from abc import ABC, abstractmethod
import numpy as np


class AbstractModelAdapter(ABC):

    @abstractmethod
    def predict_f(self, X: dict[str, float]) -> tuple[np.array, np.array]:
        """
        predict mean and standard deviation of output from input X without inclusion of noise.
        """
        pass

    @abstractmethod
    def predict_y(self, X: dict[str, float], **kwargs) -> tuple[np.array, np.array]:
        """
        predict mean and standard deviation of output from input X with inclusion of noise.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Safely close the model by releasing all CUDA resources.
        """
        pass
