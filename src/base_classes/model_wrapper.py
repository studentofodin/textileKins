from typing import List
from typing import Tuple
import pathlib as pl
import yaml
from omegaconf import DictConfig

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.base_classes.model_interface import *

class ModelWrapper(AbstractModelWrapper):

    def __init__(self, config):

        self._config = config

        self._n_models = len(self._config.usedModels)
        self._model_names = dict()
        self._model_props = dict()
        self._means = dict()
        self._vars = dict()
        self._outputs = dict()
        self._outputs_array = np.zeros((self._n_models))
        self._machine_models = dict()
        self._output_names = list()

        for model_name in self._config.usedModels:

            with open(pl.Path(self._config.pathToModels) / (model_name + '.yaml'), 'r') as stream:
                try:
                    properties = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

                output_name = properties["output"]

                if output_name in self._model_names.keys():
                    raise KeyError(f'{self._model_names[output_name]} AND {model_name} have {output_name} as output'
                                   'which leads to multiple uses of same key in dictionaries of ModelWrapper')

                else:
                    self._model_names[output_name] = model_name
                    self._model_props[output_name] = properties
                    self._means[output_name] = 0.0
                    self._vars[output_name] = 0.0
                    self._outputs[output_name] = 0.0
                    self._output_names.append(output_name)

                    model_class = properties["model_class"]
                    path_to_pkl = pl.Path(self._config.pathToModels) / (model_name + '.pkl')
                    if model_class == "SVGP":
                        mdl = AdapterSVGP(path_to_pkl, self._config.rescaleY)
                    elif model_class == "GPy_GPR":
                        mdl = AdapterGPy(path_to_pkl, self._config.rescaleY)
                    else:
                        raise (TypeError(f"The model class {model_class} is not yet supported"))
                    self._machine_models[output_name] = mdl

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def n_models(self) -> int:
        return self._n_models

    @property
    def model_names(self) -> dict:
        return self._model_names

    @property
    def model_props(self) -> dict:
        return self._model_props

    @property
    def means(self) -> dict:
        return self._means

    @property
    def vars(self) -> dict:
        return self._vars

    @property
    def outputs(self) -> dict:
        return self._outputs

    @property
    def outputs_array(self) -> np.array:
        return self._outputs_array

    @property
    def machine_models(self) -> dict:
        return self._machine_models

    @property
    def output_names(self) -> List[str]:
        return self._output_names

    def call_models(self, input: dict, latent: bool) -> None:
        if latent:
            for output_name in self._output_names:
                self._means[output_name], self._vars[output_name] = self._machine_models[output_name].predict_f(input)
        else:
            for output_name in self._output_names:
                self._means[output_name], self._vars[output_name] = self._machine_models[output_name].predict_y(input)

    def interpret_model_outputs(self) -> None:
        for i, output_name in enumerate(self._output_names):
            self._outputs[output_name] = np.random.normal(self._means[output_name], self._vars[output_name])
            self._outputs_array[i] = self._outputs[output_name]

    def get_outputs(self, input: dict, latent: bool = False) -> Tuple[np.array, dict]:
        self.call_models(input, latent)
        self.interpret_model_outputs()
        return self._outputs_array, self._outputs
