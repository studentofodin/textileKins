from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AbstractModelInterface(ABC):
    @property
    @abstractmethod
    def model_properties(self) -> dict:
        pass

    @property
    @abstractmethod
    def model(self) -> any:
        pass

    @property
    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict_y(self, X: np.array) -> np.array:
        pass

    @abstractmethod
    def predict_f(self, X: np.array) -> np.array:
        pass

    @abstractmethod
    def calc_mean_and_std(self, X: np.array, latent: bool) -> list[np.array, np.array, np.array]:
        pass