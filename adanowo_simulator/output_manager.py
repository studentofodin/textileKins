import pathlib as pl
import importlib
import sys
import yaml
from omegaconf import DictConfig
import numpy as np
import torch
import pandas as pd
import logging
from types import ModuleType

from adanowo_simulator.abstract_base_class.output_manager import AbstractOutputManager
from adanowo_simulator import model_adapter

logger = logging.getLogger(__name__)

class OutputManager(AbstractOutputManager):

    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._config = config.copy()
        self._process_models = dict()
        self._init_model_allocation()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, controls: dict[str, float], disturbances: dict[str, float]) -> dict[str, float]:
        mean_pred = dict()
        var_pred = dict()
        outputs = dict()
        if self._config.outputs_are_latent:
            for output_name, model in self._process_models.items():
                X = controls | disturbances | outputs
                mean_pred[output_name], var_pred[output_name] = model.predict_f(X)
                outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name])).item()
        else:
            for output_name, model in self._process_models.items():
                X = controls | disturbances | outputs
                mean_pred[output_name], var_pred[output_name] = \
                    model.predict_y(X, observation_noise_only=self._config.observation_noise_only)
                outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name])).item()

        return outputs

    def update_model_allocation(self, changed_outputs: list[str]) -> None:
        for output_name in changed_outputs:
            self._allocate_model_to_output(output_name, self._config.output_models[output_name])

    def reset(self) -> None:
        self._config = self._initial_config.copy()
        self._init_model_allocation()


    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        # load model properties dict from .yaml file.
        with open(pl.Path(self._config.path_to_models) / (model_name + '.yaml'), 'r') as stream:
            try:
                properties = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # load machine model.
        model_class = properties["model_class"]
        if "keep_y_scaled" in properties:
            rescale_y_temp = not bool(properties["keep_y_scaled"])
        else:
            rescale_y_temp = True

        if model_class == "Gpytorch":
            data_load = pd.read_hdf(
                pl.Path(self._config.path_to_models) / (model_name + ".hdf5"))
            if torch.cuda.is_available():
                map_location = None
            else:
                map_location = torch.device('cpu')
                logger.warning(f"No Cuda GPU found for model {model_name}. Step execution will be much slower.")
            model_state = torch.load(
                pl.Path(self._config.path_to_models) / (model_name + ".pth"), map_location=map_location)
            spec = importlib.util.spec_from_file_location("module.name",
                                                          pl.Path(self._config.path_to_models) / (
                                                                  model_name + ".py"))
            model_lib = importlib.util.module_from_spec(spec)
            sys.modules["module.name"] = model_lib
            spec.loader.exec_module(model_lib)

            mdl = model_adapter.AdapterGpytorch(model_lib, data_load, model_state, properties,
                                                  rescale_y=rescale_y_temp)

        elif model_class == "Python_script":
            model_module = self._load_model_from_script(
                pl.Path(self._config.path_to_models) / (model_name + ".py"))
            mdl = model_adapter.AdapterPyScript(model_module)

        else:
            raise (TypeError(f"The model class {model_class} is not yet supported"))

        self._process_models[output_name] = mdl

    def _load_model_from_script(self, path: pl.Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location("module.name", path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        return model_module

    def _init_model_allocation(self) -> None:
        for output_name, model_name in self._config.output_models.items():
            try:
                self._allocate_model_to_output(output_name, model_name)
                logger.info(f"Allocated model {model_name} to output {output_name}.")
            except Exception as e:
                logger.exception(f"Could not allocate model {model_name} to output {output_name}.")
                raise e


