from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class AbstractModelInterface(ABC):
    @property
    @abstractmethod
    def model_properties(self) -> dict:
        pass

    @property
    @abstractmethod
    def model(self) -> any:
        pass

    @abstractmethod
    def predict_y(self, X: dict) -> Tuple[np.array, np.array]:
        pass

    @abstractmethod
    def predict_f(self, X: dict) -> Tuple[np.array, np.array]:
        pass

    @abstractmethod
    def predict_f_internal(self, X: np.array) -> Tuple[np.array, np.array]:
        pass

    def predict_y_internal(self, X: np.array) -> Tuple[np.array, np.array]:
        pass