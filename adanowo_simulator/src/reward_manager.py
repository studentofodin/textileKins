from typing import Callable

from omegaconf import DictConfig

from src.abstract_base_class.reward_manager import AbstractRewardManager


class RewardManager(AbstractRewardManager):
    def __init__(self, config: DictConfig, reward_function: Callable):
        self._initial_config = config.copy()
        self._config = config.copy()
        self._reward_function = reward_function
        self._reward_range = (config.reward_range.lower, config.reward_range.upper)

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def reward_range(self) -> tuple[float, float]:
        return self._reward_range

    def calculate_reward(self, state: dict[str, float], outputs: dict[str, float], safety_met: bool) -> tuple[float, bool]:
        reqs_met = self._check_requirements(outputs)

        # penalty
        if not (reqs_met and safety_met):
            reward = -self._config.penalty

        # no penalty
        else:
            reward = self._reward_function(state, outputs, self._config.reward_function)

        return reward, reqs_met

    def reset(self) -> None:
        self._config = self._initial_config.copy()

    def _check_requirements(self, outputs: dict[str, float]) -> bool:
        # assume that requirement constraints are met.
        reqs_met = True

        # check simple fixed bounds for outputs.
        for output_name, lower_bound in self._config.requirements.simple_output_bounds.lower.items():
            if outputs[output_name] < lower_bound:
                reqs_met = False
        for output_name, upper_bound in self._config.requirements.simple_output_bounds.upper.items():
            if outputs[output_name] > upper_bound:
                reqs_met = False

        return reqs_met
