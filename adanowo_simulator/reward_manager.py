import importlib
import pathlib as pl

from omegaconf import DictConfig

from adanowo_simulator.abstract_base_class.reward_manager import AbstractRewardManager


class RewardManager(AbstractRewardManager):
    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._config = config.copy()
        self._reward_range = (config.reward_range.lower, config.reward_range.upper)

        # reward function.
        spec = importlib.util.spec_from_file_location("module.name",
                                                      pl.Path(self._config.path_to_reward_functions) /
                                                      (self._config.reward_function + ".py"))
        reward_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward_module)
        self.__get_reward = reward_module.get_reward

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
            reward = self._get_reward(controls, disturbances, outputs)

        return reward, output_constraints_met

    def reset(self) -> None:
        self._config = self._initial_config.copy()

    def _get_reward(self, controls: dict[str, float], disturbances: dict[str, float],
                    outputs: dict[str, float]) -> float:
        reward = self.__get_reward(controls, disturbances, outputs, self._config.reward_parameters)
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
