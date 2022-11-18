from abc import ABC, abstractmethod
import numpy as np

from experiment_tracking import AbstractExperimentTracking
from reward import AbstractReward
from interfaces import ModelInterface



class AbstractITAEnvironment(ABC):

    @property
    @abstractmethod
    def reward(self) -> AbstractReward:
        pass

    @property
    @abstractmethod
    def experimentTracker(self) -> AbstractExperimentTracking:
        pass

    @property
    @abstractmethod
    def machine(self) -> ModelInterface:
        pass

    @property
    @abstractmethod
    def maxSteps(self) -> int:
        pass

    @property
    @abstractmethod
    def currentState(self) -> np.array:
        pass
