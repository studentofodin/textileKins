from abc import ABC, abstractmethod
import numpy as np


class AbstractModelAdapter(ABC):

    @abstractmethod
    def predict_f(self, X: dict[str, float]) -> tuple[np.array, np.array]:
        pass

    @abstractmethod
    def predict_y(self, X: dict[str, float], **kwargs) -> tuple[np.array, np.array]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
