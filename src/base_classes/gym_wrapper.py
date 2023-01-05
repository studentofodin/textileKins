import numpy as np

from gym import Env
from gym.envs.registration import register

from src.base_classes.env import TrainingEnvironment


class GymWrapper(Env):

    def __init__(self, env: TrainingEnvironment, metadata: dict = None):
        self._env = env
        self._metadata = metadata

    def step(self, action: np.array):
        observation, reward, done, _, info = self._env.step(action)
        return observation, reward, done, info

    def reset(self, seed=None, options=None):
        return self._env.reset()

    def close(self):
        pass

    def render(self):
        pass

    @property
    def reward_range(self):
        return None


register(
    id='adaNowo-simulator-v0',
    entry_point='src.base_classes.gym_wrapper.GymWrapper'
)
