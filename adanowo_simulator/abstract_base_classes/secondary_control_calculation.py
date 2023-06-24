import numpy as np
from abc import ABC, abstractmethod

class AbstractSecondaryControlModel(ABC):

    @abstractmethod
    def calculate_control(self, X: dict[str, float]) -> np.array:
        pass