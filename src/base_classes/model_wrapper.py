from typing import List

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.base_classes.model_interface import *

class ModelWrapper(AbstractModelWrapper):

    def __init__(self, model_props: List[dict], model_dir: pl.Path, rescale_y: bool = True):
        self._n_models = len(model_props)
        self._means = np.zeros((self._n_models))
        self._vars = np.zeros((self._n_models))
        self._outputs_array = np.zeros((self._n_models))
        self._outputs = dict()
        self._machine_models = list()
        for mp in model_props:
            model_class = mp["model_class"]
            if model_class == "SVGP":
                mdl = AdapterSVGP(mp, model_dir, rescale_y)
            elif model_class == "GPy_GPR":
                mdl = AdapterGPy(mp, model_dir, rescale_y)
            else:
                raise (TypeError(f"The model class {model_class} is not yet supported"))
            self._machine_models.append(mdl)
            self._outputs[mdl.model_properties["output"]] = 0

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
    def outputs_array(self) -> np.array:
        return self._outputs_array

    @property
    def outputs(self) -> dict:
        return self._outputs

    def call_models(self, input: dict, latent: bool) -> None:
        means = np.zeros((self._n_models))
        vars = np.zeros((self._n_models))

        if latent:
            for i, mdl in enumerate(self._machine_models):
                means[i], vars[i] = mdl.predict_f(input)
        else:
            for i, mdl in enumerate(self._machine_models):
                means[i], vars[i] = mdl.predict_y(input)

        self._means = means
        self._vars = vars

    def interpret_model_outputs(self) -> None:
        for i in range(self._n_models):
            value = np.random.normal(self._means[i], self._vars[i])
            mdl = self._machine_models[i]
            key = mdl.model_properties["output"]
            self._outputs[key] = value
            self._outputs_array[i] = value

    def get_outputs(self, input: dict, latent: bool = False) -> dict:
        self.call_models(input, latent)
        self.interpret_model_outputs()
        return self._outputs_array, self._outputs
