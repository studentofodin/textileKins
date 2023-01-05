from abc import ABC, abstractmethod
import numpy as np


class AbstractReward(ABC):

    @property
    @abstractmethod
    def requirements(self) -> dict:
        pass

    @property
    @abstractmethod
    def rewardValue(self) -> float:
        pass

    @abstractmethod
    def calculateReward(self, currentState: np.array, currentModelOutput: np.array, safetyFlag: bool) -> float:
        pass

    @abstractmethod
    def calculatePenalty(self, state: np.array, modelOutput: np.array) -> float:
        pass
