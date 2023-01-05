from typing import Tuple
from gym import spaces
import numpy as np
from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.base_classes.reward import Reward
from src.base_classes.experiment_tracker import ExperimentTracker
from src.base_classes.model_wrapper import ModelWrapper
from src.base_classes.configuration import Configuration


class TrainingEnvironment(AbstractTrainingEnvironment):
    def __init__(self, config, machine: ModelWrapper, reward: Reward,
                 experimentTracker: ExperimentTracker, initialState: np.array):
        self._config = config
        self._machine = machine
        self._reward = reward
        self._experimentTracker = experimentTracker
        self._currentState = initialState
        self._initialState = initialState

    @property
    def machine(self):
        return self._machine

    @property
    def reward(self):
        return self._reward

    @property
    def experimentTracker(self):
        return self._experimentTracker

    @property
    def config(self):
        return self._config

    @property
    def currentState(self) -> np.array:
        return self._currentState

    @property
    def actionSpace(self) -> np.array:
        return self.actionSpace

    @property
    def observationSpace(self) -> np.array:
        return self.observationSpace

    @property
    def rewardRange(self) -> Tuple[float, float]:
        return 1.0, 100.0

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        observation = action*2
        reward = 4.0
        done = False
        info = {}
        return observation, reward, done, False, info

    def reset(self):
        self._currentState = self._initialState
