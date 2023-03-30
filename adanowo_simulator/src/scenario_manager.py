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
            if scenario[0][0] <= 0:
                raise Exception("outputModels: first step index of a scenario hast to be greater than 0."
                                f"But {output_name} scenario starts with step index {scenario[0][0]}.")
            for i in range(len(scenario)-1):
                if scenario[i][0] >= scenario[i+1][0]:
                    raise Exception("outputModels: step indices of a scenario have to be ascending."
                                    f"But for {output_name} we have"
                                    f"[{scenario[i][0]}, {scenario[i][1]}], [{scenario[i+1][0]}, {scenario[i+1][1]}]")

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def update_output_models(self, step_index: int, model_wrapper_config: DictConfig) -> List[str]:

        changed = []

        for output_name, scenario in self._config.outputModels.items():
            if scenario and scenario[0][0] == step_index:
                model_wrapper_config.outputModels[output_name] = scenario[0][1]
                changed.append(output_name)
                self._config.outputModels[output_name].pop(0)

        return changed
