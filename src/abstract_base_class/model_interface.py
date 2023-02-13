from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class AbstractModelInterface(ABC):

    @property
    @abstractmethod
    def model(self) -> any:
        pass

    @abstractmethod
    def predict_y(self, X: dict) -> Tuple[np.array, np.array]:
        """
        predict mean and variance for output y from input x.
        """
        pass

    @abstractmethod
    def predict_f(self, X: dict) -> Tuple[np.array, np.array]:
        """
        predict mean and variance for latent output f from input x.
        """
        pass

    @abstractmethod
    def predict_f_internal(self, X: np.array) -> Tuple[np.array, np.array]:
        """
        used in predict_y().
        """
        pass

    def predict_y_internal(self, X: np.array) -> Tuple[np.array, np.array]:
        """
        used in predict_f().
        """
        pass