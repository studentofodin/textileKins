from abc import ABC, abstractmethod
import numpy as np

from experiment_tracking import AbstractExperimentTracking
from reward import AbstractReward
from interfaces import AbstractModelInterface
from src.base_classes.model_wrapper_class import ModelWrapperClass



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
    def machine(self) -> ModelWrapperClass:
        pass

    @property
    @abstractmethod
    def maxSteps(self) -> int:
        pass

    @property
    @abstractmethod
    def currentState(self) -> np.array:
        pass

    @property
    @abstractmethod
    def actionSpace(self) -> np.array:
        pass

    @property
    @abstractmethod
    def observationSpace(self) -> np.array:
        pass

    @property
    @abstractmethod
    def rewardRange(self) -> list(float,float):
        pass

    @property
    @abstractmethod
    def spec(self) -> dict:
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict:
        pass

    @property
    @abstractmethod
    def npRandom(self) -> any:
        pass

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self) -> AbstractExperimentTracking:
        pass

    @abstractmethod
    def close(self):
        pass