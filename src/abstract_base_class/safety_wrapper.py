from abc import ABC, abstractmethod
import numpy as np

class AbstractSafetyWrapper(ABC):

    @property
    @abstractmethod
    def constraints(self) -> dict:
        pass

    @abstractmethod
    def isWithinConstraints(self) -> bool:
        pass

    @abstractmethod
    def calculateClippedState(self) :
        pass

    @abstractmethod
    def loadConstraints(self, configFile):
        pass
