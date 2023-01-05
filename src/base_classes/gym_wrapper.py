import numpy as np

from gym import Env
from gym.envs.registration import register

from src.base_classes.env import TrainingEnvironment


class GymWrapper(Env):
    
    def __init__(self, env: TrainingEnvironment, metadata: dict = None):
        self.env = env
        self._metadata = metadata
        self.action_space = self.env.actionSpace
        self.observation_space = self.env.observationSpace
        self._reward_range = self.env.rewardRange
   
    def step(self, action: np.array):
        """Steps through the environment with action."""
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def close(self):
        pass

    def render(self):
        pass

    @property
    def reward_range(self):
        return self._reward_range


register(
    id='adaNowo-simulator-v0',
    entry_point='src.base_classes.gym_wrapper.GymWrapper'
)
