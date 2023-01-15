import pathlib as pl
import dill
from typing import List
import numpy as np

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.base_classes.model_interface import *

class ModelWrapper(AbstractModelWrapper):

    def __init__(self, model_props: List[dict]):
        self._n_models = len(model_props)
        self._means = np.zeros((self._n_models))
        self._vars = np.zeros((self._n_models))
        self._outputs = np.zeros((self._n_models))
        self._machine_models = list()
        for mp in model_props:
            self._machine_models.append(ModelWrapper.load_model(mp))

    @property
    def machine_models(self) -> List[AbstractModelInterface]:
        return self._machine_models

    @property
    def n_models(self) -> int:
        return self._n_models

    @property
    def means(self) -> np.array:
        return self._means

    @property
    def vars(self) -> np.array:
        return self._vars

    @property
    def outputs(self) -> np.array:
        return self._outputs

    def call_models(self, input: dict, latent: bool) -> None:
        means = np.zeros((self._n_models))
        vars = np.zerso((self._n_models))

        if latent:
            for i, mdl in enumerate(self._machine_models):
                means[i], vars[i] = mdl.predict_f(input)
        else:
            for i, mdl in enumerate(self._machine_models):
                means[i], vars[i] = mdl.predict_y(input)

        self._means = means
        self._vars = vars

    def interpret_model_outputs(self) -> None:
        self._outputs = np.random.normal(self._means, self._vars)

    def get_outputs(self, input: dict, latent: bool = False) -> np.array:
        self.call_models(input, latent)
        self.interpret_model_outputs()
        return self._outputs
