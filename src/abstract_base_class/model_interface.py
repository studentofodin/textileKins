from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AbstractModelInterface(ABC):
    @property
    @abstractmethod
    def modelProperties(self) -> dict:
        pass

    @property
    @abstractmethod
    def model(self) -> any:
        pass

    @property
    @abstractmethod
    def featureImportance(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def predictY(self, X: np.array) -> np.array:
        pass

    @abstractmethod
    def predictF(self, X: np.array) -> np.array:
        pass

    @abstractmethod
    def calcMeanAndStd(self, X: np.array, latent: bool) -> list[np.array, np.array, np.array]:
        pass