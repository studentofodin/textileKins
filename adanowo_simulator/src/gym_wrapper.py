import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.registration import register
from omegaconf import DictConfig

from src.abstract_base_class.environment import AbstractTrainingEnvironment


class GymWrapper(Env):
    
    def __init__(self, env: AbstractTrainingEnvironment, config: DictConfig, metadata: dict = None):
        self._env = env
        self._metadata = metadata
        self._reward_range = self._env.reward_range
        self._config = config.copy()

        self._action_space = spaces.Box(
            low=np.array([action.low for action in self._config.action_space.values()], dtype=np.float32),
            high=np.array([action.high for action in self._config.action_space.values()], dtype=np.float32),
            shape=(len(self._config.action_space),)
        )

        self._observation_space = spaces.Box(
            low=np.array([obs.low for obs in self._config.observation_space.values()], dtype=np.float32),
            high=np.array([obs.high for obs in self._config.observation_space.values()], dtype=np.float32),
            shape=(len(self._config.observation_space),)
        )

    @property
    def env(self):
        return self._env

    @property
    def reward_range(self):
        return self._reward_range
   
    def step(self, action: np.array):
        return self._env.step(action)

    def reset(self, seed=None, options=None):
        return self._env.reset()

    def render(self):
        return self._env.render()


register(
    id='adaNowo-simulator-v0',
    entry_point='src.base_classes.gym_wrapper.GymWrapper'
)
