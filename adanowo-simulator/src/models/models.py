import numpy as np
import pandas as pd

from src.abstract_base_class.model_interface import AbstractModelInterface


class ModelA(AbstractModelInterface):
    def __init__(self):
        pass

    def modelProperties(self) -> dict:
        props = {
            "inputs": ["disturbance_d", "input_b", "input_c"],
            "output": "target_a"
        }
        return props

    @property
    def model(self) -> any:
        return []

    @property
    def featureImportance(self) -> pd.DataFrame:
        return pd.DataFrame()

    def predictY(self, X: np.array) -> np.array:
        return np.array(X[0]*X[1]-X[2])

    def predictF(self, X: np.array) -> np.array:
        return np.array(X[0]*X[1]-X[2])

    def calcMeanAndStd(self, X: np.array, latent: bool) -> list[np.array, np.array, np.array]:
        mean = np.array(X[0]*X[1]-X[2])
        std = 1.2
        upper = mean - std
        lower = mean + std
        return [mean, upper, lower]  # the model wrapper needs to sample from this gaussian distribution to
        # produce a value for the reward function


class ModelB(AbstractModelInterface):
    def __init__(self):
        pass

    def modelProperties(self) -> dict:
        props = {
            "inputs": ["input_d", "input_c"],
            "output": "target_b"
        }
        return props

    @property
    def model(self) -> any:
        return []

    @property
    def featureImportance(self) -> pd.DataFrame:
        return pd.DataFrame()

    def predictY(self, X: np.array) -> np.array:
        return np.array(X[0]+X[1])

    def predictF(self, X: np.array) -> np.array:
        return np.array(X[0]+X[1])

    def calcMeanAndStd(self, X: np.array, latent: bool) -> list[np.array, np.array, np.array]:
        mean = np.array(X[0]+X[1])
        std = 1.7
        upper = mean - std
        lower = mean + std
        return [mean, upper, lower]  # the model wrapper needs to sample from this gaussian distribution
