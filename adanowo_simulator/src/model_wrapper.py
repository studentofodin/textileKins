import pathlib as pl
import yaml
from omegaconf import DictConfig
import numpy as np
from typing import List

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.model_interface import AbstractModelInterface
from src import model_interface


class ModelWrapper(AbstractModelWrapper):

    def __init__(self, config: DictConfig):
        self._initialconfig = config.copy()
        self._n_models = len(config.outputModels)

        self._machine_models = dict()
        self.reset()

    @property
    def machine_models(self) -> dict[str, AbstractModelInterface]:
        return self._machine_models

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def get_outputs(self, input_model: dict[str, float]) -> tuple[np.array, dict]:
        mean_pred, var_pred = self._call_models(input_model)
        outputs_array, outputs = self._interpret_model_outputs(mean_pred, var_pred)
        return outputs_array, outputs

    def update(self, changed_outputs: List[str]) -> None:
        for output_name in changed_outputs:
            self._allocate_model_to_output(output_name, self._config.outputModels[output_name])

    def reset(self) -> None:
        self._config = self._initialconfig.copy()
        for output_name, model_name in self._config.outputModels.items():
            self._allocate_model_to_output(output_name, model_name)


    def _call_models(self, input_model: dict[str, float], latent=False) -> (dict[str, float], dict[str, float]):
        mean_pred = dict()
        var_pred = dict()
        if latent:
            for output_name, model in self._machine_models:
                mean_pred[output_name], var_pred[output_name] = \
                    model.predict_f(input_model)
        else:
            for output_name, model in self._machine_models.items():
                mean_pred[output_name], var_pred[output_name] = \
                    model.predict_y(input_model)
        return mean_pred, var_pred

    def _interpret_model_outputs(self, mean_pred: dict[str, float], var_pred: dict[str, float]) \
            -> (np.array, dict[str, float]):
        outputs = dict()
        outputs_array = np.zeros(self._n_models)
        for i, output_name in enumerate(self._config.outputModels.keys()):
            outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name]))
            outputs_array[i] = outputs[output_name]
        return outputs_array, outputs

    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        # load model properties dict from .yaml file.
        with open(pl.Path(self._config.pathToModels) / (model_name + '.yaml'), 'r') as stream:
            try:
                properties = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        if output_name != properties["output"]:
            raise Exception(f"output name argument ({output_name}) does not match output name from model properties ({properties['output']}.")

        # load machine model.
        model_class = properties["model_class"]
        path_to_pkl = pl.Path(self._config.pathToModels) / properties["model_path"]
        if model_class == "SVGP":
            mdl = model_interface.AdapterSVGP(path_to_pkl, True)
        elif model_class == "GPy_GPR":
            mdl = model_interface.AdapterGPy(path_to_pkl, True)
        else:
            raise (TypeError(f"The model class {model_class} is not yet supported"))
        self._machine_models[output_name] = mdl



