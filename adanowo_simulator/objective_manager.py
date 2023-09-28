from typing import Callable
from omegaconf import DictConfig, OmegaConf

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

    def step(self, state: dict[str, float], outputs: dict[str, float],
             setpoint_constraints_met: dict[str, bool], dependent_variable_constraints_met: dict[str, bool]) -> \
            tuple[float, dict[str, bool]]:
        if self._ready:
            output_constraints_met = self._output_constraints_met(outputs)
            # penalty.
            if not (all(
                    (output_constraints_met | setpoint_constraints_met | dependent_variable_constraints_met).values())):
                reward = -self._get_penalty(state, outputs)
            # no penalty.
            else:
                reward = self._get_reward(state, outputs)
        else:
            raise Exception("Cannot call step() before calling reset().")
        return reward, output_constraints_met

    def reset(self, initial_state: dict[str, float], initial_outputs: dict[str, float],
              setpoint_constraints_met_initially: dict[str, bool],
              dependent_variable_constraints_met_initially: dict[str, bool]) -> tuple[float, dict[str, bool]]:
        self._config = self._initial_config.copy()
        self._ready = True
        reward, output_constraints_met = self.step(
            initial_state, initial_outputs, setpoint_constraints_met_initially,
            dependent_variable_constraints_met_initially)
        return reward, output_constraints_met

    def close(self) -> None:
        self._ready = False

    def _get_reward(self, state: dict[str, float], outputs: dict[str, float]) -> float:
        reward = self._reward_function(state, outputs, self._config.reward_parameters)
        return reward

    def _get_penalty(self, state: dict[str, float], outputs: dict[str, float]) -> float:
        penalty = self._penalty_function(state, outputs, self._config.reward_parameters)
        return penalty

    def _output_constraints_met(self, outputs: dict[str, float]) -> dict[str, bool]:
        output_constraints_met = dict()
        for output_name, bounds in self._config.output_bounds.items():
            if "lower" in bounds.keys():
                if outputs[output_name] < bounds.lower:
                    output_constraints_met[f"{output_name}.lower"] = False
                else:
                    output_constraints_met[f"{output_name}.lower"] = True
            if "upper" in bounds.keys():
                if outputs[output_name] > bounds.upper:
                    output_constraints_met[f"{output_name}.upper"] = False
                else:
                    output_constraints_met[f"{output_name}.upper"] = True

        return output_constraints_met
