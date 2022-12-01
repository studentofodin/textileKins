import numpy as np
import pandas as pd

from abstract_base_class.model_interface import AbstractModelInterface

class ModelInterface(AbstractModelInterface):
    def __init__(self, model_properties, model=True, feature_importance=pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})):
        self._model_properties=model_properties
        self._model=model
        self._feature_importance=feature_importance

    @property
    def model_properties(self):
        return self._model_properties

    @property
    def model(self):
        return self._model

    @property
    def feature_importance(self):
        return self.feature_importance

    def predict_y(self, X: np.array) -> np.array:
        return np.ones(10)

    def predict_f(self, X:np.array):
        return np.ones(3)

    def calc_mean_and_std(self, X: np.array, latent: bool) -> list[np.array, np.array, np.array]:
        return [np.ones(3), np.ones(3), np.ones(3)]

