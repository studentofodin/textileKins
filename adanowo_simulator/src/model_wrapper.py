import pathlib as pl

import yaml
from omegaconf import DictConfig
import numpy as np

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.model_interface import AbstractModelInterface
from src import model_interface


class ModelWrapper(AbstractModelWrapper):

    def __init__(self, config: DictConfig):

        self._config = config

        # all dictionary properties have output names as keys.
        self._n_models = len(self._config.usedModels)
        self._model_names = dict()
        self._model_props = dict()
        self._machine_models = dict()
        self._output_names = list()
        self._scenario_idxs = dict()

        # fill properties.
        for output_name, model_name in self._config.usedModels.items():
            self._allocate_model_to_output(output_name, model_name, True)

        self.reset_scenario()




    @property
    def n_models(self) -> int:
        return self._n_models

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    @property
    def model_names(self) -> dict[str, str]:
        return self._model_names

    @property
    def model_props(self) -> dict[str, any]:
        return self._model_props

    @property
    def machine_models(self) -> dict[str, AbstractModelInterface]:
        return self._machine_models

    def _call_models(self, input_model: dict[str, float], latent=False) -> (dict[str, float], dict[str, float]):
        mean_pred = dict()
        var_pred = dict()
        if latent:
            for output_name in self._output_names:
                mean_pred[output_name], var_pred[output_name] = \
                    self._machine_models[output_name].predict_f(input_model)
        else:
            for output_name in self._output_names:
                mean_pred[output_name], var_pred[output_name] = \
                    self._machine_models[output_name].predict_y(input_model)
        return mean_pred, var_pred

    def _interpret_model_outputs(self, mean_pred: dict[str, float], var_pred: dict[str, float]) \
            -> (np.array, dict[str, float]):
        outputs = dict()
        outputs_array = np.zeros(self._n_models)
        for i, output_name in enumerate(self._output_names):
            outputs[output_name] = np.random.normal(mean_pred[output_name], np.sqrt(var_pred[output_name]))
            outputs_array[i] = outputs[output_name]
        return outputs_array, outputs

    def get_outputs(self, input_model: dict[str, float]) -> tuple[np.array, dict]:
        mean_pred, var_pred = self._call_models(input_model)
        outputs_array, outputs = self._interpret_model_outputs(mean_pred, var_pred)
        return outputs_array, outputs

    def _allocate_model_to_output(self, output_name: str, model_name: str, initial_flag: bool) -> None:

        # load model properties dict from .yaml file.
        with open(pl.Path(self._config.pathToModels) / (model_name + '.yaml'), 'r') as stream:
            try:
                properties = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        if output_name != properties["output"]:
            raise Exception("output name argument does not match output name from model properties")

        if initial_flag:
            # check if output_name is not yet a model output.
            if output_name in self._output_names:
                raise Exception(f"{output_name} would be used twice as a model output which is not possible.")
            self._output_names.append(output_name)

        else:
            # check if output_name is already a model output.
            if output_name not in self._output_names:
                raise Exception(f"{output_name} would be a new model output which is not possible in this state.")

        self._model_names[output_name] = model_name
        self._model_props[output_name] = properties

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

    def update(self, step_index: int) -> None:
        for output_name, scenario in self._config.scenario.models.items():
            scenario_idx = self._scenario_idxs[output_name]
            if (scenario_idx < len(scenario)) and (step_index == scenario[scenario_idx][0]):
                self._allocate_model_to_output(output_name, scenario[scenario_idx][1], False)
                self._scenario_idxs[output_name] += 1

    def reset_scenario(self) -> None:
        for output_name in self._config.scenario.models.keys():
            self._scenario_idxs[output_name] = 0



