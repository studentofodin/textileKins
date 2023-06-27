import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
from gymnasium.envs.registration import register
from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.environment import AbstractEnvironment

class GymWrapper(Env):

    def __init__(self, environment: AbstractEnvironment, config: DictConfig):
        self._environment: AbstractEnvironment = environment
        self._config: DictConfig = config.copy()

        self._action_space: spaces.Box = spaces.Box(
            low=np.array([action_name.low for action_name in self._config.action_space.values()], dtype=np.float32),
            high=np.array([action_name.high for action_name in self._config.action_space.values()], dtype=np.float32)
        )

        self._observation_space: spaces.Box = spaces.Box(
            low=np.array([obs_name.low for obs_name in self._config.observation_space.values()], dtype=np.float32),
            high=np.array([obs_name.high for obs_name in self._config.observation_space.values()], dtype=np.float32)
        )

    @property
    def environment(self) -> AbstractEnvironment:
        return self._environment

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space
   
    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        observation, reward = self._environment.step(action)
        return observation, reward, False, False, dict()

    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        super().reset(seed=seed)
        observation, _ = self._environment.reset()
        return observation, dict()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self) -> None:
        self._environment.close()


register(
    id='adaNowo-simulator-v0',
    entry_point='src.base_classes.gym_wrapper.GymWrapper'
)
