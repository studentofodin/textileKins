from typing import Callable

from omegaconf import DictConfig

from src.abstract_base_class.reward_manager import AbstractRewardManager


class RewardManager(AbstractRewardManager):
    def __init__(self, config: DictConfig, reward_function: Callable):
        self._initial_config = config.copy()
        self._config = None
        self._reward_function = reward_function
        self._reward_range = (config.reward_range.lower, config.reward_range.upper)
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def reward_range(self) -> tuple[float, float]:
        return self._reward_range

    def step(self, controls: dict[str, float], disturbances: dict[str, float], outputs: dict[str, float],
             control_constraints_met: bool) -> tuple[float, bool]:
        output_constraints_met = self._output_constraints_met(outputs)

        # penalty.
        if not (output_constraints_met and control_constraints_met):
            reward = -self._config.penalty

        # no penalty.
        else:
            reward = self._reward_function(controls, disturbances, outputs, self._config.reward_function)

        return reward, output_constraints_met

    def reset(self) -> None:
        self._config = self._initial_config.copy()

    def _output_constraints_met(self, outputs: dict[str, float]) -> bool:
        # assume that output constraints are met.
        output_constraints_met = True

        # check simple fixed bounds for outputs.
        for output_name, lower_bound in self._config.output_bounds.lower.items():
            if outputs[output_name] < lower_bound:
                output_constraints_met = False
        for output_name, upper_bound in self._config.output_bounds.upper.items():
            if outputs[output_name] > upper_bound:
                output_constraints_met = False

        return output_constraints_met
