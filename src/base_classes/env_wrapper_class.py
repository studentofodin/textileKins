from gym import Env, spaces
import numpy as np
from base_classes.env_class import ITAEnvironment

class ITAEnvWrapper(Env):

    def __init__(self):
        self._ITAEnv = ITAEnvironment()
        self.action_space = spaces.Box(np.array([0,1,2],dtype=np.float32), np.array([100,100,100],dtype=np.float32))
        self.observation_space = spaces.Box(np.array([1,1,1],dtype=np.float32), np.array([1000,1000,900],dtype=np.float32))
        self.spec = None
        self.metadata = {'Render Mode' : 'W&B', 'RenderX':30}
        self.np_random = np.random.default_rng()

    def step(self, action:np.array):
        return self.ITAEnv._step(action)

    def reset(self):
        return self.ITAEnv._reset()

    def render(self):
        return self.ITAEnv._render()

    def close(self):
        pass

    @property
    def ITAEnv(self):
        return self._ITAEnv

    # @ITAEnv.setter
    # def ITAEnv(self, ITAEnv):
    #     self._ITAEnv = ITAEnv

    @property
    def action_space(self):
        return self.ITAEnv.actionSpace

    @action_space.setter
    def action_space(self, action_space):
        self.ITAEnv.actionSpace = action_space

    @property
    def observation_space(self):
        return self.ITAEnv.observationSpace

    @observation_space.setter
    def observation_space(self, observation_space):
        self.ITAEnv.observationSpace = observation_space

    @property
    def reward_range(self):
        return self.ITAEnv.rewardRange

    @reward_range.setter
    def reward_range(self, reward_range):
        self.ITAEnv.rewardRange = reward_range

    @property
    def spec(self):
        return self.ITAEnv.spec

    @spec.setter
    def spec(self, spec):
        self.ITAEnv.spec = spec

    @property
    def metadata(self):
        return self.ITAEnv.metadata

    @metadata.setter
    def metadata(self, metadata):
        self.ITAEnv.metadata = metadata

    @property
    def np_random(self):
        return self.ITAEnv.npRandom

    @np_random.setter
    def np_random(self, np_random):
        self.ITAEnv.npRandom = np_random

    

