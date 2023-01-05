import numpy as np
import pandas as pd

from src.abstract_base_class.model_interface import AbstractModelInterface


class ModelInterface(AbstractModelInterface):
    def __init__(self, modelProperties: dict, model=True,
                 featureImportance=pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})):
        self._modelProperties = modelProperties
        self._model = model
        self._featureImportance = featureImportance

    @property
    def modelProperties(self):
        return self._modelProperties

    @property
    def model(self):
        return self._model

    @property
    def featureImportance(self):
        return self.featureImportance

    def predictY(self, X: np.array) -> np.array:
        return np.array(X[0]+X[1])

    def predictF(self, X: np.array):
        return np.array(X[0]+X[1])

    def calcMeanAndStd(self, X: np.array, latent: bool) -> list[np.array, np.array, np.array]:
        mean = np.array(X[0]+X[1])
        std = 1.7
        upper = mean + std
        lower = mean - std
        return [mean, upper, lower]  # the model wrapper needs to sample from this gaussian distribution