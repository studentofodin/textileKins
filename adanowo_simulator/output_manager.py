import importlib
import pathlib as pl
import logging
import sys
from multiprocessing import Process, Pipe
import yaml
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch

from adanowo_simulator.abstract_base_classes.model_adapter import AbstractModelAdapter
from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator import model_adapter

logger = logging.getLogger(__name__)
RECEIVE = 0
SEND = 1
DEFAULT_RELATIVE_PATH = "./output_models"


def model_executor(mdl: AbstractModelAdapter, input_pipe: Pipe, output_pipe: Pipe, model_uncertainty_only: bool):
    while True:
        if input_pipe.poll():
            input_recv = input_pipe.recv()
            if input_recv is None:
                mdl.close()
                break
            if model_uncertainty_only:
                mean_pred, var_pred = mdl.predict_f(input_recv)
            else:
                mean_pred, var_pred = mdl.predict_y(input_recv, observation_noise_only=True)
            output_pipe.send((mean_pred, var_pred))


class ParallelOutputManager(AbstractOutputManager):

    def __init__(self, config: DictConfig):
        # use default path
        main_script_path = pl.Path(__file__).resolve().parent
        self._path_to_output_models = main_script_path.parent / DEFAULT_RELATIVE_PATH

        if config.path_to_output_models is not None:
            temp_path = pl.Path(config.path_to_models)
            if self._path_to_output_models.is_dir():
                logger.info(f"Using custom output model path {self._path_to_output_models}.")
                self._path_to_output_models = temp_path
            else:
                raise Exception(
                    f"Custom output model path {self._path_to_output_models} is not valid.")

        # Add model path to sys.path so that the models can be imported.
        sys.path.append(str(self._path_to_output_models))

        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = self._initial_config.copy()
        self._model_processes: dict[str, Process] = dict()
        self._model_config: DictConfig = OmegaConf.create()
        self._input_pipes: dict[str, Pipe] = dict()
        self._output_pipes: dict[str, Pipe] = dict()
        self._model_uncertainty_only: bool = False
        self._ready = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, state: dict[str, float]) -> dict[str, float]:
        if self._ready:
            self._update_model_allocation()
            try:
                mean_pred, var_pred = self._call_models(state)
                outputs = self._sample_output_distribution(mean_pred, var_pred)
            except Exception as e:
                self.close()
                raise e
        else:
            raise Exception("Cannot call step() before calling reset().")
        return outputs

    def reset(self, state: dict[str, float]) -> dict[str, float]:
        self.close()
        self._config = self._initial_config.copy()
        self._model_config = self._config.output_models.copy()
        for output_name, model_name in self._config.output_models.items():
            try:
                self._allocate_model_to_output(output_name, model_name)
            except Exception as e:
                self.close()
                raise e
        self._ready = True
        outputs = self.step(state)
        return outputs

    def close(self) -> None:
        if self._model_processes:
            for output_name in self._model_processes.keys():
                if self._model_processes[output_name].is_alive():
                    self._input_pipes[output_name][SEND].send(None)
                    self._model_processes[output_name].join()
                    self._input_pipes[output_name][SEND].close()
            self._model_processes = dict()
            self._input_pipes = dict()
            self._output_pipes = dict()
            self._ready = False

    def _update_model_allocation(self) -> None:
        for output_name, model_name in self._config.output_models.items():
            if self._model_config[output_name] != model_name:
                try:
                    self._input_pipes[output_name][SEND].send(None)
                    self._model_processes[output_name].join()
                    self._input_pipes[output_name][SEND].close()
                    self._model_config[output_name] = model_name
                    self._allocate_model_to_output(output_name, model_name)
                except Exception as e:
                    self.close()
                    raise e

    def _call_models(self, X: dict[str, float]) -> (dict[str, np.array], dict[str, np.array]):
        mean_pred = dict()
        var_pred = dict()

        for output_name in self._model_processes.keys():
            self._input_pipes[output_name][SEND].send(X)
        for output_name in self._model_processes.keys():
            mean_pred[output_name], var_pred[output_name] = self._output_pipes[output_name][RECEIVE].recv()
        return mean_pred, var_pred

    def _sample_output_distribution(self, mean_pred: dict[str, np.array], var_pred: dict[str, np.array]) \
            -> dict[str, float]:
        outputs = dict()
        for output_name in self._config.output_models.keys():
            outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name])).item()
        return outputs

    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        # load model properties dict from .yaml file
        with open(self._path_to_output_models / (model_name + '.yaml'), 'r') as stream:
            properties = yaml.safe_load(stream)

        model_class = properties["model_class"]

        if model_class == "Gpytorch":
            if "keep_y_scaled" in properties:
                rescale_y_temp = not bool(properties["keep_y_scaled"])
            else:
                rescale_y_temp = True
            data_load = pd.read_hdf(
                self._path_to_output_models / (model_name + ".hdf5")
            )
            if torch.cuda.is_available():
                map_location = None
            else:
                map_location = torch.device('cpu')
                logger.warning(f"No Cuda GPU found for model {model_name}. Step execution will be much slower.")
            model_state = torch.load(
                self._path_to_output_models / (model_name + ".pth"), map_location=map_location
            )
            importlib.import_module(model_name)
            model_module = sys.modules[model_name]

            mdl = model_adapter.AdapterGpytorch(model_module, data_load, model_state, properties,
                                                rescale_y=rescale_y_temp)
        elif model_class == "Python_script":
            importlib.import_module(model_name)
            model_module = sys.modules[model_name]
            mdl = model_adapter.AdapterPyScript(model_module)

        else:
            raise (TypeError(f"The model class {model_class} is not yet supported"))

        new_input_pipe = Pipe()
        new_output_pipe = Pipe()
        self._input_pipes[output_name] = new_input_pipe
        self._output_pipes[output_name] = new_output_pipe
        self._model_processes[output_name] = \
            Process(target=model_executor, args=(mdl, new_input_pipe[RECEIVE], new_output_pipe[SEND],
                                                 True, self._model_uncertainty_only))
        self._model_processes[output_name].start()
        logger.info(f"Allocated model {model_name} to output {output_name}.")


class SequentialOutputManager(AbstractOutputManager):

    def __init__(self, config: DictConfig):
        # use default path
        main_script_path = pl.Path(__file__).resolve().parent
        self._path_to_output_models = main_script_path.parent / DEFAULT_RELATIVE_PATH

        if config.path_to_output_models is not None:
            temp_path = pl.Path(config.path_to_models)
            if self._path_to_output_models.is_dir():
                logger.info(f"Using custom output model path {self._path_to_output_models}.")
                self._path_to_output_models = temp_path
            else:
                raise Exception(
                    f"Custom output model path {self._path_to_output_models} is not valid.")

        # Add model path to sys.path so that the models can be imported.
        sys.path.append(str(self._path_to_output_models))

        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = self._initial_config.copy()
        self._output_models: dict[str, AbstractModelAdapter] = dict()
        self._model_config: DictConfig = OmegaConf.create()
        self._model_uncertainty_only: bool = False
        self._ready = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, state: dict[str, float]) -> dict[str, float]:
        if self._ready:
            self._update_model_allocation()
            mean_pred, var_pred = self._call_models(state)
            outputs = self._sample_output_distribution(mean_pred, var_pred)
        else:
            raise Exception("Cannot call step() before calling reset().")
        return outputs

    def reset(self, initial_state: dict[str, float]) -> dict[str, float]:
        self._config = self._initial_config.copy()
        self._model_config = self._config.output_models.copy()
        for output_name, model_name in self._config.output_models.items():
            self._allocate_model_to_output(output_name, model_name)
        self._ready = True
        outputs = self.step(initial_state)
        return outputs

    def close(self) -> None:
        self._ready = False

    def _update_model_allocation(self) -> None:
        for output_name, model_name in self._config.output_models.items():
            if self._model_config[output_name] != model_name:
                self._model_config[output_name] = model_name
                self._allocate_model_to_output(output_name, model_name)

    def _call_models(self, X: dict[str, float]) -> (dict[str, np.array], dict[str, np.array]):
        mean_pred = dict()
        var_pred = dict()

        for output_name, mdl in self._output_models.items():
            if self._model_uncertainty_only:
                mean_pred[output_name], var_pred[output_name] = mdl.predict_f(X)
            else:
                mean_pred[output_name], var_pred[output_name] = mdl.predict_y(
                    X, observation_noise_only=True)

        return mean_pred, var_pred

    def _sample_output_distribution(self, mean_pred: dict[str, np.array], var_pred: dict[str, np.array]) \
            -> dict[str, float]:
        outputs = dict()
        for output_name in self._config.output_models.keys():
            outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name])).item()
        return outputs

    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        # load model properties dict from .yaml file
        with open(self._path_to_output_models / (model_name + '.yaml'), 'r') as stream:
            properties = yaml.safe_load(stream)

        model_class = properties["model_class"]

        if model_class == "Gpytorch":
            if "keep_y_scaled" in properties:
                rescale_y_temp = not bool(properties["keep_y_scaled"])
            else:
                rescale_y_temp = True
            data_load = pd.read_hdf(
                self._path_to_output_models / (model_name + ".hdf5")
            )
            if torch.cuda.is_available():
                map_location = None
            else:
                map_location = torch.device('cpu')
                logger.warning(f"No Cuda GPU found for model {model_name}. Step execution will be much slower.")
            model_state = torch.load(
                self._path_to_output_models / (model_name + ".pth"), map_location=map_location
            )
            importlib.import_module(model_name)
            model_module = sys.modules[model_name]

            mdl = model_adapter.AdapterGpytorch(model_module, data_load, model_state, properties,
                                                rescale_y=rescale_y_temp)
        elif model_class == "Python_script":
            importlib.import_module(model_name)
            model_module = sys.modules[model_name]
            mdl = model_adapter.AdapterPyScript(model_module)

        else:
            raise (TypeError(f"The model class {model_class} is not yet supported"))

        self._output_models[output_name] = mdl
        logger.info(f"Allocated model {model_name} to output {output_name}.")
