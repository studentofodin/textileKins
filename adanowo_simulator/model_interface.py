from typing import OrderedDict
from copy import copy
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from gpytorch.models import ExactGP
from torch.cuda import is_available as cuda_is_available
from torch.cuda import FloatTensor as CudaFloatTensor
from torch import FloatTensor
from torch import squeeze as torch_squeeze
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

<<<<<<<< HEAD:adanowo_simulator/model_interface.py
from adanowo_simulator.abstract_base_class.model_interface import AbstractModelInterface
========
from src.abstract_base_class.model_adapter import AbstractModelInterface
from src.abstract_base_class.python_script_model import AbstractPyScriptModule
>>>>>>>> review-ruben:adanowo_simulator/model_adapters.py


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer is a dummy that does nothing to its input.
    It is used to make the pipeline work without a final estimator.
    """
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    @staticmethod
    def transform(self, input_array, y=None):
        return input_array * 1


class AdapterGpytorch(AbstractModelInterface):

    def __init__(self, model_module: ExactGP, data: pd.DataFrame, model_state: OrderedDict, model_properties: dict,
                 rescale_y: bool = True) -> None:
        self._unpack_func = model_module.unpack_dict
        self._properties = model_properties
        self._rescale_y = rescale_y
        self._scaler_y = None

        # load the internal data transformation pipeline
        x_numpy = data[self._properties["training_inputs"]].to_numpy()
        y_numpy = data[self._properties["training_target"]].to_numpy().reshape(-1, 1)
        list_transform = []
        if self._properties["X_is_scaled"]:
            list_transform.append(
                ("scaler", RobustScaler())
            )
        if self._properties["pca_on_inputs"]:
            list_transform.append(
                ("pca", PCA())
            )
        list_transform.append(
            ("identity", IdentityTransformer())
        )
        self._pipe = Pipeline(list_transform)
        self._pipe = self._pipe.fit(x_numpy)

        if self._properties["y_is_scaled"]:
            self._scaler_y = RobustScaler().fit(y_numpy)
            y_numpy = self._scaler_y.transform(y_numpy)

        # construct the model and load state from dict
        if cuda_is_available():
            self._Tensor = CudaFloatTensor
            x_tensor = self._numpy_to_model_input(x_numpy)
            y_tensor = torch_squeeze(self._Tensor(
                y_numpy
            ))
            self._likelihood = model_module.likelihood.cuda()
            self._model = model_module.ExactGPModel(x_tensor, y_tensor, self._likelihood).cuda()
        else:
            self._Tensor = FloatTensor
            x_tensor = self._numpy_to_model_input(x_numpy)
            y_tensor = torch_squeeze(self._Tensor(
                y_numpy
            ))
            self._likelihood = model_module.likelihood
            self._model = model_module.ExactGPModel(x_tensor, y_tensor, self._likelihood)

        self._model.load_state_dict(model_state)
        self._model.eval()
        self._likelihood.eval()

        noise_var_scaled = self._likelihood.noise.cpu().detach().numpy().reshape(-1, 1)
        _, self._noise_variance = self._rescaler_y(noise_var_scaled, noise_var_scaled)

    def _numpy_to_model_input(self, x_temp: np.array) -> FloatTensor:
        tensor_out = self._Tensor(
            self._pipe.transform(
                x_temp
            )
        )
        return tensor_out

    def _rescaler_y(self, y_temp, var_temp):
        if self._scaler_y is not None and self._rescale_y:
            y_temp = self._scaler_y.inverse_transform(y_temp)
            var_temp = var_temp * np.power(self._scaler_y.scale_[0], 2)
        return y_temp, var_temp

    def _predict_f_internal(self, X: np.array) -> [np.array, np.array]:
        x_tensor = self._numpy_to_model_input(X)
        f_pred = self._model(x_tensor)
        f_pred, var = f_pred.mean.cpu().detach().numpy().reshape(-1, 1), \
            f_pred.variance.cpu().detach().numpy().reshape(-1, 1)
        f_pred, var = self._rescaler_y(f_pred, var)
        return f_pred, var

    def _predict_y_internal(self, X: np.array) -> [np.array, np.array]:
        x_tensor = self._numpy_to_model_input(X)
        y_pred = self._likelihood(self._model(x_tensor))
        y_pred, var = y_pred.mean.cpu().detach().numpy().reshape(-1, 1), \
            y_pred.variance.cpu().detach().numpy().reshape(-1, 1)
        y_pred, var = self._rescaler_y(y_pred, var)
        return y_pred, var

    def predict_f(self, X: dict) -> np.array:
        X = self._unpack_func(X, self._properties["training_inputs"])
        y_pred, var = self._predict_f_internal(X)
        return y_pred, var

    def predict_y(self, X: dict, **kwargs) -> np.array:
        X = self._unpack_func(X, self._properties["training_inputs"])
        y_pred, var = self._predict_y_internal(X)
        if "observation_noise_only" in kwargs:
            if kwargs["observation_noise_only"]:
                var_scalar = copy(self._noise_variance)
                var = np.ones_like(y_pred) * var_scalar
        else:
            ValueError("Nope")
        return y_pred, var


class AdapterPyScript(AbstractModelInterface):

    def __init__(self, model_module: ModuleType) -> None:
        self._model = model_module.model

    def predict_f(self, X: dict) -> np.array:
        f_pred, var = self._model(X)
        var = np.zeros_like(var)
        return f_pred, var

    def predict_y(self, X: dict, **kwargs) -> np.array:
        f_pred, var = self._model(X)
        return f_pred, var
