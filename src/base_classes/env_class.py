from typing import Tuple
from gym import spaces
import numpy as np
from abstract_base_class.environment import AbstractITAEnvironment
from base_classes.reward_class import Reward
from base_classes.experiment_tracker_class import ExperimentTracker
from base_classes.model_wrapper_class import ModelWrapper
from base_classes.configuration_class import Configuration


class ITAEnvironment(AbstractITAEnvironment):
    def __init__(self):

        self.config = Configuration({'a':1,'b':3},{'c':4},{'d':7},{'e':6},{'f':4},500,{'g':5})
        self.machine = ModelWrapper()
        self.reward = Reward({"key1": 2, "key2": 3})
        self.experimentTracker = ExperimentTracker({"key5": 3, "key6": 6})
        self.currentState = np.ones(3)

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

    @property
    def currentState(self) -> np.array:
        return self._currentState

    @currentState.setter
    def currentState(self, currentState):
        self._currentState = currentState

    @property
    def actionSpace(self) -> np.array:
        return self.actionSpace

    @actionSpace.setter
    def actionSpace(self, actionSpace):
        self._actionSpace = actionSpace

    @property
    def observationSpace(self) -> np.array:
        return self.observationSpace

    @observationSpace.setter
    def observationSpace(self, observationSpace):
        self._observationSpace = observationSpace

    @property
    def rewardRange(self) -> Tuple[float, float]:
        return (1.0,100.0)

    @rewardRange.setter
    def rewardRange(self, rewardRange):
        self._rewardRange = rewardRange

    @property
    def spec(self) -> dict:
        return {'specKey1':3,'specKey2':9}

    @spec.setter
    def spec(self, spec):
        self._spec = spec
    
    @property
    def metadata(self) -> dict:
        return {'metaKey1':8,'metaKey2':9}
    
    @metadata.setter
    def metadata(self, metadata):
        self._metadata = metadata

    @property
    def npRandom(self) -> any:
        return 9.0
    
    @npRandom.setter
    def npRandom(self, npRandom):
        self._npRandom = npRandom

    def _step(self, action:np.array) -> Tuple[np.array, float, bool, bool, dict]:
        return Tuple[np.ones(3), 4.0, True, False, {}]

    def _reset(self) -> Tuple[np.array, dict]:
        return Tuple[np.zeros(3),{}]

    def _render(self) -> any:
        print("Rendering")

    def _close(self):
        print("Closing")
