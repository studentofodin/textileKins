from abc import ABC, abstractmethod

import numpy as np


class AbstractModelInterface(ABC):


    @abstractmethod
    def predict_f(self, X: dict[str, float]) -> tuple[np.array, np.array]:
        """
        predict mean and standard deviation of output from input X without inclusion of noise.
        """
        pass

    @abstractmethod
    def predict_y(self, X: dict[str, float]) -> tuple[np.array, np.array]:
        """
        predict mean and standard deviation of output from input X with inclusion of noise.
        """
        pass

    @abstractmethod
    def _predict_f_internal(self, X: np.array) -> tuple[np.array, np.array]:
        """
        predict mean and standard deviation of output from input X without inclusion of noise.
        input X is np.array instead of dict[str, float].
        """
        pass

    def _predict_y_internal(self, X: np.array) -> tuple[np.array, np.array]:
        """
        predict mean and standard deviation of output from input X with inclusion of noise.
        input X is np.array instead of dict[str, float].
        """
        pass
