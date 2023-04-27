import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.registration import register
from omegaconf import DictConfig

from src.abstract_base_class.environment import AbstractTrainingEnvironment


class GymWrapper(Env):
    
    def __init__(self, env: AbstractTrainingEnvironment, config: DictConfig):
        self._env = env
        self._config = config.copy()

        self._action_space = spaces.Box(
            low=np.array([action_name.low for action_name in self._config.action_space.values()], dtype=np.float32),
            high=np.array([action_name.high for action_name in self._config.action_space.values()], dtype=np.float32),
            shape=(len(self._config.action_space),)
        )

        self._observation_space = spaces.Box(
            low=np.array([obs_name.low for obs_name in self._config.observation_space.values()], dtype=np.float32),
            high=np.array([obs_name.high for obs_name in self._config.observation_space.values()], dtype=np.float32),
            shape=(len(self._config.observation_space),)
        )

    @property
    def env(self) -> AbstractTrainingEnvironment:
        return self._env

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space
   
    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        return self._env.step(action)

    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        super().reset(seed=seed)
        return self._env.reset()



register(
    id='adaNowo-simulator-v0',
    entry_point='src.base_classes.gym_wrapper.GymWrapper'
)
