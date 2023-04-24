import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.registration import register
from omegaconf import DictConfig

from src.environment import TrainingEnvironment


class GymWrapper(Env):
    
    def __init__(self, env: TrainingEnvironment, config: DictConfig, metadata: dict = None):
        self.env = env
        self._metadata = metadata
        self._reward_range = self.env.rewardRange
        self._config = config

        self._actionSpace = spaces.Box(
            low=np.array([action.low for action in self._config.actionSpace.values()], dtype=np.float32),
            high=np.array([action.high for action in self._config.actionSpace.values()], dtype=np.float32),
            shape=(len(self._config.actionSpace),)
        )

        self._observationSpace = spaces.Box(
            low=np.array([obs.low for obs in self._config.observationSpace.values()], dtype=np.float32),
            high=np.array([obs.high for obs in self._config.observationSpace.values()], dtype=np.float32),
            shape=(len(self._config.observationSpace),)
        )

    @property
    def reward_range(self):
        return self._reward_range
   
    def step(self, action: np.array):
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def render(self):
        return self.env.render()


register(
    id='adaNowo-simulator-v0',
    entry_point='src.base_classes.gym_wrapper.GymWrapper'
)
