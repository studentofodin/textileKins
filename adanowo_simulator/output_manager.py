import importlib
import pathlib as pl
import logging
import sys
from multiprocessing import Process, Pipe

import yaml
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch

from adanowo_simulator.abstract_base_classes.model_adapter import AbstractModelAdapter
from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator import model_adapter

logger = logging.getLogger(__name__)
RECEIVE = 0
SEND = 1
DEFAULT_RELATIVE_PATH = "./models"


def model_executor(mdl: AbstractModelAdapter, input_pipe: Pipe, output_pipe: Pipe, latent=False):
    while True:
        if input_pipe.poll():
            input_recv = input_pipe.recv()
            if input_recv is None:
                mdl.shutdown()
                break
            if latent:
                mean_pred, var_pred = mdl.predict_f(input_recv)
            else:
                mean_pred, var_pred = mdl.predict_y(input_recv, observation_noise_only=True)
            output_pipe.send((mean_pred, var_pred))


class OutputManager(AbstractOutputManager):

    def __init__(self, config: DictConfig):
        # use default path
        main_script_path = pl.Path(__file__).resolve().parent
        self._model_path = main_script_path.parent / DEFAULT_RELATIVE_PATH

        if config.path_to_models is not None:
            temp_path = pl.Path(config.path_to_models)
            if self._model_path.is_dir():
                logger.info(f"Using custom model path {self._model_path}.")
                self._model_path = temp_path

        # Add model path to sys.path so that the models can be imported.
        sys.path.append(str(self._model_path))

        self._initial_config = config.copy()
        self._config = config.copy()
        self._model_processes = dict()
        self._input_pipe = dict()
        self._output_pipe = dict()
        self.init_model_allocation()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, inputs: dict[str, float]) -> dict[str, float]:
        mean_pred, var_pred = self._call_models(inputs)
        outputs = self._sample_output_distribution(mean_pred, var_pred)
        return outputs

    def init_model_allocation(self) -> None:
        for output_name, model_name in self._config.output_models.items():
            try:
                self._allocate_model_to_output(output_name, model_name)
                logger.info(f"Allocated model {model_name} to output {output_name}.")
            except Exception as e:
                logger.exception(f"Could not allocate model {model_name} to output {output_name}.")
                raise e

    def update_model_allocation(self, changed_outputs: list[str]) -> None:
        for changed_output in changed_outputs:
            self._allocate_model_to_output(changed_output, self._config.output_models[changed_output])

    def start_processes(self) -> None:
        for output_name, model_process in self._model_processes.items():
            model_process.start()
            logger.info(f"Process for output {output_name} is running and listening for inputs")

    def shutdown(self) -> None:
        logger.info("Waiting for all processes to finish...")
        for _, input_pipe in self._input_pipe.items():
            input_pipe[SEND].send(None)
        for _, model_process in self._model_processes.items():
            model_process.join()
        for _, input_pipe in self._input_pipe.items():
            input_pipe[SEND].close()

    def reset(self) -> None:
        self._config = self._initial_config.copy()
        self.shutdown()
        logger.info("All processes finished. Reallocating...")
        self.init_model_allocation()

    def _call_models(self, inputs: dict[str, float], latent=False) -> (dict[str, np.array], dict[str, np.array]):
        mean_pred = dict()
        var_pred = dict()

        for output_name, model_process in self._model_processes.items():
            if not model_process.is_alive():
                model_process.start()
                logger.info(f"Process for output {output_name} is running and listening for inputs")
            self._input_pipe[output_name][SEND].send(inputs)

        for output_name, model_process in self._model_processes.items():
            mean_pred[output_name], var_pred[output_name] = self._output_pipe[output_name][RECEIVE].recv()
        return mean_pred, var_pred

    def _sample_output_distribution(self, mean_pred: dict[str, np.array], var_pred: dict[str, np.array]) \
            -> dict[str, float]:
        outputs = dict()
        for output_name in self._config.output_models.keys():
            outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name])).item()
        return outputs

    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        # load model properties dict from .yaml file
        with open(self._model_path / (model_name + '.yaml'), 'r') as stream:
            try:
                properties = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.exception(f"Could not load model properties for model {model_name}.")
                raise exc

        # load process model
        model_class = properties["model_class"]
        if "keep_y_scaled" in properties:
            rescale_y_temp = not bool(properties["keep_y_scaled"])
        else:
            rescale_y_temp = True

        if model_class == "Gpytorch":
            data_load = pd.read_hdf(
                self._model_path / (model_name + ".hdf5")
            )
            if torch.cuda.is_available():
                map_location = None
            else:
                map_location = torch.device('cpu')
                logger.warning(f"No Cuda GPU found for model {model_name}. Step execution will be much slower.")
            model_state = torch.load(
                self._model_path / (model_name + ".pth"), map_location=map_location
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
        self._input_pipe[output_name] = new_input_pipe
        self._output_pipe[output_name] = new_output_pipe
        self._model_processes[output_name] = \
            Process(target=model_executor, args=(mdl, new_input_pipe[RECEIVE], new_output_pipe[SEND], False))
