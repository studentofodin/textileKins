from abc import ABC, abstractmethod
import numpy as np


class AbstractPyScriptModule(ABC):
    @abstractmethod
    def model(self, X: dict[str, float]) -> tuple[np.array, np.array]:
        """
        predict mean and standard deviation of output from input X with inclusion of noise.
        """
        pass
