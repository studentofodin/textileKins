from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from abstract_base_class.experiment_tracker import AbstractExperimentTracker
from abstract_base_class.reward import AbstractReward
from abstract_base_class.model_wrapper import AbstractModelWrapper



class AbstractITAEnvironment(ABC):

    @property
    @abstractmethod
    def reward(self) -> AbstractReward:
        pass

    @property
    @abstractmethod
    def experimentTracker(self) -> AbstractExperimentTracker:
        pass

    @property
    @abstractmethod
    def machine(self) -> AbstractModelWrapper:
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
    def rewardRange(self) -> Tuple[float,float]:
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
    def _step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        pass

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _render(self) -> any:
        pass

    @abstractmethod
    def _close(self):
        pass