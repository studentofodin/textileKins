from gym import Env
import numpy as np
from base_classes.env_class import ITAEnvironment

class ITAEnvWrapper(Env):

    def __init__(self, action={'action1':9,'action2':5}):
        self.ITAEnv = ITAEnvironment()
        self._action = action
        pass

    def step(self):
        return self.ITAEnv._step(self._action)

    def reset(self):
        return self.ITAEnv._reset()

    def render(self):
        return self.ITAEnv._render()

    def close(self):
        pass

    @property
    def ITAEnv(self):
        return self._ITAEnv

    @ITAEnv.setter
    def ITAEnv(self, ITAEnv):
        self._ITAEnv = ITAEnv

    @property
    def action_space(self):
        return self.ITAEnv._actionSpace()

    @property
    def observation_space(self):
        return self.ITAEnv._observationSpace

    @property
    def reward_range(self):
        return self.ITAEnv._rewardRange()

    @property
    def spec(self):
        return self.ITAEnv._spec()

    @property
    def metadata(self):
        return self.ITAEnv._metadata()

    @property
    def np_random(self):
        return self.ITAEnv._npRandom()

    

