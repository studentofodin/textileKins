from types import MethodType
from typing import Callable

from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.abstract_base_classes.reward_manager import AbstractRewardManager

class RewardManager(AbstractRewardManager):
    def __init__(self, reward_function: Callable, config: DictConfig):
        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = OmegaConf.create()
        self._reward_function: MethodType = reward_function
        self._ready: bool = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, state: dict[str, float], outputs: dict[str, float], control_constraints_met: bool) \
            -> tuple[float, bool]:
        if self._ready:
            output_constraints_met = self._output_constraints_met(outputs)

            # penalty.
            if not (output_constraints_met and control_constraints_met):
                reward = -self._config.penalty
            # no penalty.
            else:
                reward = self._get_reward(state, outputs)

        else:
            raise Exception("Cannot call step() before calling reset().")

        return reward, output_constraints_met


    def reset(self, state: dict[str, float], outputs: dict[str, float], control_constraints_met: bool) -> \
            tuple[float, bool]:
        self._config = self._initial_config.copy()
        self._ready = True
        reward, output_constraints_met = self.step(state, outputs, control_constraints_met)
        return reward, output_constraints_met

    def _get_reward(self, state: dict[str, float], outputs: dict[str, float]) -> float:
        reward = self._reward_function(state, outputs, self._config.reward_parameters)
        return reward

    def _output_constraints_met(self, outputs: dict[str, float]) -> bool:
        # assume that output constraints are met.
        output_constraints_met = True

        for output_name, bounds in self._config.output_bounds.items():
            if "lower" in bounds.keys():
                if outputs[output_name] < bounds.lower:
                    output_constraints_met = False
            if "upper" in bounds.keys():
                if outputs[output_name] > bounds.upper:
                    output_constraints_met = False

        return output_constraints_met
