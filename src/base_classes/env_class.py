from gym import spaces
import numpy as np
from abstract_base_class.environment import AbstractITAEnvironment
from reward_class import Reward
from experiment_tracker_class import ExperimentTracker
from model_wrapper_class import ModelWrapper
# from configuration_class import Configuration


class ITAEnvironment(AbstractITAEnvironment):
    def __init__(self, maxSteps=200, current_state=np.ones(3)):

        # self.config = Configuration()
        self.machine = ModelWrapper({"key3": 7, "key4": 8})
        self.reward = Reward({"key1": 2, "key2": 3})
        self.experimentTracker = ExperimentTracker({"key5": 3, "key6": 6})
        self.maxSteps = maxSteps
        # self._current_state = current_state
        # self.action_space = spaces.Box(self.low, self.high, np.dtype)
        # self.observation_space
        # self.spec
        # self.metadata
        # self.np_random

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, modelWrapper):
        self._machine = modelWrapper

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, reward):
        self._reward = reward

    @property
    def experimentTracker(self):
        return self._experimentTracker

    @experimentTracker.setter
    def experimentTracker(self, experimentTracker):
        self._experimentTracker = experimentTracker

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    def _step(self, action:np.array) -> tuple[np.array, float, bool, bool, dict]:
        return tuple[np.ones(3), 4.0, True, False, {}]

    def _reset(self) -> tuple[np.array, dict]:
        return tuple[np.zeros(3),{}]

    def _render(self):
        pass

    def _close(self):
        pass
