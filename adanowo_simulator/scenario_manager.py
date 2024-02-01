from omegaconf import ListConfig, DictConfig
import numpy as np

from adanowo_simulator.abstract_base_classes.scenario_manager import AbstractScenarioManager
from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.objective_manager import AbstractObjectiveManager
from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager


class ScenarioManager(AbstractScenarioManager):

    def __init__(self, config: DictConfig):
        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = self._initial_config.copy()
        self._ready: bool = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, step_index: int, disturbance_manager: AbstractDisturbanceManager,
             output_manager: AbstractOutputManager, objective_manager: AbstractObjectiveManager):
        if self._ready:
            self._update_disturbances(step_index, disturbance_manager.config.disturbances)
            self._update_output_bounds(step_index, objective_manager.config.output_bounds)
            self._update_output_model_allocation(step_index, output_manager.config.output_models)
        else:
            raise RuntimeError("Cannot call step() before calling reset().")

    def reset(self) -> None:
        self._config = self._initial_config.copy()
        self._ready = True

    def close(self) -> None:
        self._ready = False

    def _update_output_model_allocation(self, step_index: int, output_models_config: DictConfig):
        if self._config.output_models is None:
            return
        for output_name, scenario in self._config.output_models.items():
            self._update_target(step_index, output_models_config, output_name, scenario)

    def _update_output_bounds(self, step_index: int, output_bounds_config: DictConfig) -> None:
        if self._config.output_bounds is None:
            return
        for output_name, scenarios in self._config.output_bounds.items():
            for boundary_type in ["lower", "upper"]:
                scenario_value = scenarios.get(boundary_type)
                if scenario_value is None:
                    continue
                self._update_target(step_index, output_bounds_config[output_name], boundary_type, scenario_value)

    def _update_disturbances(self, step_index: int, disturbance_config: DictConfig) -> None:
        if self._config.disturbances is None:
            return
        for disturbance_name, scenario in self._config.disturbances.items():
            self._update_target(step_index, disturbance_config, disturbance_name, scenario)

    @staticmethod
    def _update_target(step_index: int, target_config: DictConfig, target_field: str,
                       scenario: ListConfig | DictConfig) -> None:
        if isinstance(scenario, ListConfig):  # deterministic scenario
            if scenario and scenario[0][0] == step_index:
                target_config[target_field] = scenario[0][1]
                scenario.pop(0)
        else:  # random scenario
            if step_index % scenario.trigger_interval == 0:
                target_config[target_field] = np.random.uniform(
                    scenario.mean - scenario.range,
                    scenario.mean + scenario.range
                )
        return
