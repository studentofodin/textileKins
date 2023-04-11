from omegaconf import DictConfig
from typing import List
import numpy as np

from src.abstract_base_class.scenario_manager import AbstractScenarioManager

class ScenarioManager(AbstractScenarioManager):

    def __init__(self, config: DictConfig):
        self._initialconfig = config.copy()
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def update_output_models(self, step_index: int, output_models_config: DictConfig) -> List[str]:
        changed = []

        for output, scenario in self._config.outputModels.items():
            if scenario and scenario[0][0] == step_index:
                output_models_config[output] = scenario[0][1]
                changed.append(output)
                self._config.outputModels[output].pop(0)

        return changed

    def update_requirements(self, step_index: int, requirements_config: DictConfig) -> None:

        for output, scenario in self._config.requirements.simpleOutputBounds.lower.items():
            if scenario and scenario[0][0] == step_index:
                requirements_config.simpleOutputBounds.lower[output] = scenario[0][1]
                self._config.requirements.simpleOutputBounds.lower[output].pop(0)

        for output, scenario in self._config.requirements.simpleOutputBounds.upper.items():
            if scenario and scenario[0][0] == step_index:
                requirements_config.simpleOutputBounds.upper[output] = scenario[0][1]
                self._config.requirements.simpleOutputBounds.upper[output].pop(0)

        for req, scenario in self._config.requirements.complexConstraints.items():
            if scenario and scenario[0][0] == step_index:
                requirements_config.complexConstraints[req] = scenario[0][1]
                self._config.requirements.complexConstraints[req].pop(0)

    def update_disturbances(self, step_index: int, disturbance_config: DictConfig) -> List[str]:
        changed = []

        for disturbance, scenario in self._config.disturbances.items():
            if step_index % scenario.timeStep == 0:
                disturbance_config[disturbance] = np.random.normal(scenario.mean, scenario.std)
                changed.append(disturbance)

        return changed

    def reset(self) -> None:
        self._config = self._initialconfig.copy()

