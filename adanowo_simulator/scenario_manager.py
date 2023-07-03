from omegaconf import ListConfig, DictConfig, OmegaConf
import numpy as np

from adanowo_simulator.abstract_base_classes.scenario_manager import AbstractScenarioManager
from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.reward_manager import AbstractRewardManager
from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager


class ScenarioManager(AbstractScenarioManager):

    def __init__(self, config: DictConfig):
        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = OmegaConf.create()
        self._ready: bool = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, step_index: int, disturbance_manager: AbstractDisturbanceManager,
             output_manager: AbstractOutputManager, reward_manager: AbstractRewardManager):
        if self._ready:
            self._update_disturbances(step_index, disturbance_manager.config.disturbances)
            self._update_output_bounds(step_index, reward_manager.config.output_bounds)
            changed_outputs = self._update_output_model_allocation(step_index, output_manager.config.output_models)
            output_manager.update_model_allocation(changed_outputs)
        else:
            raise Exception("Cannot call step() before calling reset().")

    def reset(self) -> None:
        self._config = self._initial_config.copy()
        self._ready = True

    def _update_output_model_allocation(self, step_index: int, output_models_config: DictConfig) -> list[str]:
        changed_outputs = []
        for output_name, scenario in self._config.output_models.items():
            did_update = self._update_target(step_index, output_models_config, output_name, scenario)
            if did_update:
                changed_outputs.append(output_name)
        return changed_outputs

    def _update_output_bounds(self, step_index: int, output_bounds_config: DictConfig) -> None:
        for output_name, scenarios in self._config.output_bounds.items():
            if "lower" in scenarios.keys():
                scenario = scenarios.lower
                self._update_target(step_index, output_bounds_config[output_name], "lower", scenario)
            if "upper" in scenarios.keys():
                scenario = scenarios.upper
                self._update_target(step_index, output_bounds_config[output_name], "upper", scenario)

    def _update_disturbances(self, step_index: int, disturbance_config: DictConfig) -> None:
        for disturbance_name, scenario in self._config.disturbances.items():
            self._update_target(step_index, disturbance_config, disturbance_name, scenario)

    @staticmethod
    def _update_target(step_index: int, target_config: DictConfig, target_field: str,
                       scenario: ListConfig | DictConfig) -> bool:
        did_update = False
        # deterministic scenario
        if isinstance(scenario, ListConfig):
            if scenario and scenario[0][0] == step_index:
                target_config[target_field] = scenario[0][1]
                scenario.pop(0)
                did_update = True
        # random scenario
        else:
            if (step_index-1) % scenario.trigger_interval == 0:
                target_config[target_field] = np.random.normal(scenario.mean, scenario.std)
                did_update = True
        return did_update
