from abc import ABC, abstractmethod
import numpy as np


class AbstractSafetyWrapper(ABC):

    @property
    @abstractmethod
    def constraints(self) -> dict:
        pass

    @abstractmethod
    def isWithinConstraints(self, actions: dict) -> bool:
        pass

    @abstractmethod
    def calculateClippedState(self) -> np.array:
        pass

