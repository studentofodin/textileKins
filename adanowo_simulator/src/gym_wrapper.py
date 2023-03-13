import numpy as np

from gymnasium import Env, spaces
from gymnasium.envs.registration import register
from omegaconf import DictConfig

from src.env import TrainingEnvironment


class GymWrapper(Env):
    
    def __init__(self, env: TrainingEnvironment, config: DictConfig, metadata: dict = None):
        self.env = env
        self._metadata = metadata
        self._reward_range = self.env.rewardRange
        self._config = config
        # set action space.
        self._actionSpace = spaces.Box(
            low=np.array([self._config.actionSpace[param].low for
                          param in self._config.usedControls], dtype=np.float32),
            high=np.array([self._config.actionSpace[param].high for
                           param in self._config.usedControls], dtype=np.float32),
            shape=(len(self._config.usedControls),)
        )
        # set observation space.
        self._observationSpace = spaces.Box(
            low=np.array([self._config.observationSpace[param].low for param in self.env._machine.output_names],
                         dtype=np.float32),
            high=np.array([self._config.observationSpace[param].high for param in self.env._machine.output_names],
                          dtype=np.float32),
            shape=(len(self.env._machine.output_names),)
        )
   
    def step(self, action: np.array):
        """Steps through the environment with action."""
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def render(self):
        return self.env.render()

    @property
    def reward_range(self):
        return self._reward_range


register(
    id='adaNowo-simulator-v0',
    entry_point='src.base_classes.gym_wrapper.GymWrapper'
)
