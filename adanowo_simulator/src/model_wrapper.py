import pathlib as pl
import importlib
import sys

import yaml
from omegaconf import DictConfig
import numpy as np
import torch
import pandas as pd

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src import model_interface


class ModelWrapper(AbstractModelWrapper):

    def __init__(self, config: DictConfig):
        self._initialConfig = config.copy()
        self._n_outputs = len(config.outputModels)

        self._machine_models = dict()
        self.reset()

        self._config = None

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    def get_outputs(self, inputs: dict[str, float]) -> tuple[np.array, dict[str, float]]:
        mean_pred, std_pred = self._call_models(inputs)
        outputs_array, outputs = self._sample_output_distribution(mean_pred, std_pred)
        return outputs_array, outputs

    def update(self, changed_outputs: list[str]) -> None:
        for output_name in changed_outputs:
            self._allocate_model_to_output(output_name, self._config.outputModels[output_name])

    def reset(self) -> None:
        self._config = self._initialConfig.copy()
        for output_name, model_name in self._config.outputModels.items():
            self._allocate_model_to_output(output_name, model_name)

    def _call_models(self, inputs: dict[str, float], latent=False) -> (dict[str, np.array], dict[str, np.array]):
        mean_pred = dict()
        var_pred = dict()
        if latent:
            for output_name, model in self._machine_models.items():
                mean_pred[output_name], var_pred[output_name] = \
                    model.predict_f(inputs)
        else:
            for output_name, model in self._machine_models.items():
                mean_pred[output_name], var_pred[output_name] = \
                    model.predict_y(inputs)
        return mean_pred, var_pred

    def _sample_output_distribution(self, mean_pred: dict[str, np.array], var_pred: dict[str, np.array]) \
            -> (np.array, dict[str, float]):
        outputs = dict()
        for output_name in self._config.outputModels.keys():
            outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name])).item()
        outputs_array = np.array(tuple(outputs.values()))
        return outputs_array, outputs

    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        # load model properties dict from .yaml file.
        with open(pl.Path(self._config.pathToModels) / (model_name + '.yaml'), 'r') as stream:
            try:
                properties = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # load machine model.
        model_class = properties["model_class"]

        if model_class == "Gpytorch":
            data_load = pd.read_hdf(
                pl.Path(self._config.pathToModels) / (model_name + ".hdf5")
            )
            if torch.cuda.is_available():
                map_location = None
            else:
                map_location = torch.device('cpu')
                Warning("No Cuda-enabled GPU found! Code execution will still work, but is significantly slower.")
            model_state = torch.load(
                pl.Path(self._config.pathToModels) / (model_name + ".pth"), map_location=map_location
            )
            spec = importlib.util.spec_from_file_location("module.name",
                                                          pl.Path(self._config.pathToModels) / (
                                                                  model_name + ".py"))
            model_lib = importlib.util.module_from_spec(spec)
            sys.modules["module.name"] = model_lib
            spec.loader.exec_module(model_lib)

            mdl = model_interface.AdapterGpytorch(model_lib, data_load, model_state, properties, rescale_y=True)
        elif model_class == "SVGP":
            path_to_pkl = pl.Path(self._config.pathToModels) / properties["model_path"]
            mdl = model_interface.AdapterSVGP(path_to_pkl, True)
        elif model_class == "GPy_GPR":
            path_to_pkl = pl.Path(self._config.pathToModels) / properties["model_path"]
            mdl = model_interface.AdapterGPy(path_to_pkl, True)
        else:
            raise (TypeError(f"The model class {model_class} is not yet supported"))
        self._machine_models[output_name] = mdl
