from typing import Callable
from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.objective_manager import AbstractObjectiveManager


class ObjectiveManager(AbstractObjectiveManager):
    def __init__(self, objective_function: Callable, penalty_function: Callable, config: DictConfig):
        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = self._initial_config.copy()
        self._reward_function = objective_function
        self._penalty_function = penalty_function
        self._ready: bool = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, state: dict[str, float], outputs: dict[str, float], setpoints_okay: dict[str, bool],
             dependent_variables_okay: dict[str, bool]) -> tuple[float, dict[str, bool]]:
        if self._ready:
            outputs_okay = self._output_constraints_satisfied(outputs)
            if not (all(
                    (outputs_okay | setpoints_okay | dependent_variables_okay).values())):
                reward = self._get_penalty(state, outputs)  # penalty
            else:
                reward = self._get_reward(state, outputs)  # no penalty
        else:
            raise Exception("Cannot call step() before calling reset().")
        return reward, outputs_okay

    def reset(self, initial_state: dict[str, float], initial_outputs: dict[str, float],
              setpoints_okay_initially: dict[str, bool],
              dependent_variables_okay_initially: dict[str, bool]) -> tuple[float, dict[str, bool]]:
        self._config = self._initial_config.copy()
        self._ready = True
        reward, output_constraints_met = self.step(
            initial_state, initial_outputs, setpoints_okay_initially,
            dependent_variables_okay_initially)
        return reward, output_constraints_met

    def close(self) -> None:
        self._ready = False

    def _get_reward(self, state: dict[str, float], outputs: dict[str, float]) -> float:
        reward = self._reward_function(state, outputs, self._config.reward_parameters)
        return reward

    def _get_penalty(self, state: dict[str, float], outputs: dict[str, float]) -> float:
        penalty = self._penalty_function(state, outputs, self._config.reward_parameters)
        return penalty

    def _output_constraints_satisfied(self, outputs: dict[str, float]) -> dict[str, bool]:
        output_constraints_met = dict()
        for output_name, boundaries in self._config.output_bounds.items():
            for boundary_type in ["lower", "upper"]:
                boundary_value = boundaries.get(boundary_type)
                if boundary_value is None:
                    continue
                comparison = outputs[output_name] <= boundary_value if boundary_type == "lower" else (
                        outputs[output_name] >= boundary_value)
                output_constraints_met[f"{output_name}.{boundary_type}"] = not comparison

        return output_constraints_met
