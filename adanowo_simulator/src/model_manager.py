import pathlib as pl
from importlib import util as importlib_utils
import yaml
from omegaconf import DictConfig
import numpy as np
import torch
import pandas as pd
import logging

from src.abstract_base_class.model_manager import AbstractModelManager
from src import model_adapters


logger = logging.getLogger(__name__)


def load_model_from_script(path: pl.Path) -> any:
    spec = importlib_utils.spec_from_file_location("module.name", path)
    model_module = importlib_utils.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module


class ModelManager(AbstractModelManager):

    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._config = config.copy()
        self._process_models = dict()
        self.init_model_allocation()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def get_model_outputs(self, inputs: dict[str, float]) -> tuple[np.array, dict[str, float]]:
        mean_pred, var_pred = self._call_models(inputs)
        outputs_array, outputs = self._sample_output_distribution(mean_pred, var_pred)
        return outputs_array, outputs

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

    def reset(self) -> None:
        self._config = self._initial_config.copy()
        self.init_model_allocation()

    def _call_models(self, inputs: dict[str, float], latent=False) -> (dict[str, np.array], dict[str, np.array]):
        mean_pred = dict()
        var_pred = dict()
        if latent:
            for output_name, model in self._process_models.items():
                mean_pred[output_name], var_pred[output_name] = \
                    model.predict_f(inputs)
        else:
            for output_name, model in self._process_models.items():
                mean_pred[output_name], var_pred[output_name] = \
                    model.predict_y(inputs, observation_noise_only=True)
        return mean_pred, var_pred

    def _sample_output_distribution(self, mean_pred: dict[str, np.array], var_pred: dict[str, np.array]) \
            -> (np.array, dict[str, float]):
        outputs = dict()
        for output_name in self._config.output_models.keys():
            outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name])).item()
        outputs_array = np.array(tuple(outputs.values()))
        return outputs_array, outputs

    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        # load model properties dict from .yaml file
        with open(pl.Path(self._config.path_to_models) / (model_name + '.yaml'), 'r') as stream:
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
                pl.Path(self._config.path_to_models) / (model_name + ".hdf5")
            )
            if torch.cuda.is_available():
                map_location = None
            else:
                map_location = torch.device('cpu')
                logger.warning(f"No Cuda GPU found for model {model_name}. Step execution will be much slower.")
            model_state = torch.load(
                pl.Path(self._config.path_to_models) / (model_name + ".pth"), map_location=map_location
            )
            model_module = load_model_from_script(
                pl.Path(self._config.path_to_models) / (model_name + ".py")
            )

            mdl = model_adapters.AdapterGpytorch(model_module, data_load, model_state, properties,
                                                 rescale_y=rescale_y_temp)
        elif model_class == "Python_script":
            model_module = load_model_from_script(
                pl.Path(self._config.path_to_models) / (model_name + ".py")
            )
            mdl = model_adapters.AdapterPyScript(model_module, properties)

        else:
            raise (TypeError(f"The model class {model_class} is not yet supported"))
        self._process_models[output_name] = mdl
