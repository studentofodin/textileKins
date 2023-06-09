from omegaconf import DictConfig
import numpy as np

from src.abstract_base_class.scenario_manager import AbstractScenarioManager


class ScenarioManager(AbstractScenarioManager):

    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._config = None
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def update_output_models(self, step_index: int, output_models_config: DictConfig) -> tuple[DictConfig, list[str]]:
        changed = []

        for output_name, scenario in self._config.output_models.items():
            if scenario and scenario[0][0] == step_index:
                output_models_config[output_name] = scenario[0][1]
                changed.append(output_name)
                self._config.output_models[output_name].pop(0)

        return output_models_config, changed

    def update_output_bounds(self, step_index: int, output_bounds_config: DictConfig) -> DictConfig:

        for output_name, scenarios in self._config.output_bounds.items():
            if "lower" in scenarios.keys():
                scenario = scenarios.lower
                if scenario and scenario[0][0] == step_index:
                    output_bounds_config[output_name]["lower"] = scenario[0][1]
                    self._config.output_bounds[output_name]["lower"].pop(0)
            if "upper" in scenarios.keys():
                scenario = scenarios.upper
                if scenario and scenario[0][0] == step_index:
                    output_bounds_config[output_name]["upper"] = scenario[0][1]
                    self._config.output_bounds[output_name]["upper"].pop(0)

        return output_bounds_config

    def update_disturbances(self, step_index: int, disturbance_config: DictConfig) -> DictConfig:
        for disturbance_name, scenario in self._config.disturbances.items():
            if step_index % scenario.trigger_interval == 0:
                disturbance_config[disturbance_name] = np.random.normal(scenario.mean, scenario.std)

        return disturbance_config

    def reset(self) -> None:
        self._config = self._initial_config.copy()
