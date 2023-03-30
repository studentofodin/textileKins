import pathlib as pl
import yaml
from omegaconf import DictConfig
import numpy as np
from typing import List

from src.abstract_base_class.scenario_manager import AbstractScenarioManager


class ScenarioManager(AbstractScenarioManager):

    def __init__(self, config: DictConfig, model_wrapper_config: DictConfig):
        # model_wrapper_config.outputModels["unevenness_card_web"] = "unevenness_card_web1"
        self._config = config

        # check if output model scenarios are valid.
        for output_name, scenario in self._config.outputModels.items():
            self._check_simple_scenario(scenario, "outputModels", output_name)

        # check if requirement scenarios are valid.
        for output_name, scenario in self._config.requirements.simpleOutputBounds.lower.items():
            self._check_simple_scenario(scenario, "requirements/simpleOutputBounds/lower", output_name)
        for output_name, scenario in self._config.requirements.simpleOutputBounds.upper.items():
            self._check_simple_scenario(scenario, "requirements/simpleOutputBounds/upper", output_name)
        for name, scenario in self._config.requirements.complexConstraints.items():
            self._check_simple_scenario(scenario, "requirements/simpleOutputBounds/upper", output_name)




    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def update_output_models(self, step_index: int, output_models_config: DictConfig) -> List[str]:

        changed = []

        for output_name, scenario in self._config.outputModels.items():
            if scenario and scenario[0][0] == step_index:
                output_models_config[output_name] = scenario[0][1]
                changed.append(output_name)
                self._config.outputModels[output_name].pop(0)

        return changed

    def update_requirements(self, step_index: int, requirements_config: DictConfig) -> None:

        for output_name, scenario in self._config.requirements.simpleOutputBounds.lower.items():
            if scenario and scenario[0][0] == step_index:
                requirements_config.simpleOutputBounds.lower.output_name = scenario[0][1]
                self._config.requirements.simpleOutputBounds.lower[output_name].pop(0)

        for output_name, scenario in self._config.requirements.simpleOutputBounds.upper.items():
            if scenario and scenario[0][0] == step_index:
                requirements_config.simpleOutputBounds.upper[output_name] = scenario[0][1]
                self._config.requirements.simpleOutputBounds.upper[output_name].pop(0)

        for name, scenario in self._config.requirements.complexConstraints.items():
            if scenario and scenario[0][0] == step_index:
                requirements_config.complexConstraints[name] = scenario[0][1]
                self._config.requirements.complexConstraints[name].pop(0)



    def _check_simple_scenario(self, scenario, parent_name, name):
        if scenario[0][0] <= 0:
            raise Exception(f"{parent_name}: first step index of a scenario has to be greater than 0."
                            f"But {name} scenario starts with step index {scenario[0][0]}.")
        for i in range(len(scenario) - 1):
            if scenario[i][0] >= scenario[i + 1][0]:
                raise Exception(f"{parent_name}: step indices of a scenario have to be ascending."
                                f"But for {name} we have"
                                f"[{scenario[i][0]}, {scenario[i][1]}], [{scenario[i + 1][0]}, {scenario[i + 1][1]}]")

