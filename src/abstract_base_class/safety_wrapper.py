from abc import ABC, abstractmethod
import numpy as np


class AbstractSafetyWrapper(ABC):

    @property
    @abstractmethod
    def constraints(self) -> dict:
        pass

    @abstractmethod
    def isWithinConstraints(self, state: dict) -> bool:
        pass

