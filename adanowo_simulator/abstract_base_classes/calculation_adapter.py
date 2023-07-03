import numpy as np
from abc import ABC, abstractmethod


class AbstractCalculationAdapter(ABC):

    @abstractmethod
    def calculate(self, X: dict[str, float]) -> np.array:
        pass
