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
    def calculateReward(self, currentState: dict, currentModelOutput: np.array, safetyFlag: bool) -> float:
        pass

    @abstractmethod
    def calculatePenalty(self, state: dict, modelOutput: np.array) -> float:
        pass
