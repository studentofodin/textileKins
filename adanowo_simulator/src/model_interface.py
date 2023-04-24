from typing import OrderedDict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from gpytorch.models import ExactGP
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from src.abstract_base_class.model_interface import AbstractModelInterface

CUDA_GPU_AVAILABLE = torch.cuda.is_available()


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

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
        if CUDA_GPU_AVAILABLE:
            self._Tensor = torch.cuda.FloatTensor
            x_tensor = self._numpy_to_model_input(x_numpy)
            y_tensor = torch.squeeze(self._Tensor(
                y_numpy
            ))
            self._likelihood = model_module.likelihood.cuda()
            self._model = model_module.ExactGPModel(x_tensor, y_tensor, self._likelihood,
                                                    model_properties["lengthscale_constraint"]).cuda()
        else:
            self._Tensor = torch.FloatTensor
            x_tensor = self._numpy_to_model_input(x_numpy)
            y_tensor = torch.squeeze(self._Tensor(
                y_numpy
            ))
            self._likelihood = model_module.likelihood
            self._model = model_module.ExactGPModel(x_tensor, y_tensor, self._likelihood,
                                                    model_properties["lengthscale_constraint"])
        self._model.load_state_dict(model_state)
        self._model.eval()
        self._likelihood.eval()

    def _numpy_to_model_input(self, x_temp: np.array) -> torch.FloatTensor:
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

    def predict_y(self, X: dict) -> np.array:
        X = self._unpack_func(X, self._properties["training_inputs"])
        y_pred, var = self._predict_y_internal(X)
        return y_pred, var


# class AdapterSVGP(AbstractModelInterface):
#
#     def __init__(self, path_to_pkl: pl.Path, rescale_y: bool = True) -> None:
#         with open(path_to_pkl, "rb") as file:
#             pickle_obj = dill.load(file)
#         self._model = pickle_obj["model"]
#         self._unpack_func = pickle_obj["unpack_dict_func"]
#         self._scaler_X = pickle_obj["scaler_X"]
#         self._pca_X = pickle_obj["pca_X"]
#         self._scaler_y = pickle_obj["scaler_y"]
#         list_transform = []
#         if self._scaler_X is not None:
#             list_transform.append(("scaler_X", self._scaler_X))
#         if self._pca_X is not None:
#             list_transform.append(("pca_X", self._pca_X))
#         self._pipe = Pipeline(list_transform)
#         self._rescale_y = rescale_y
#
#     @property
#     def model(self):
#         return deepcopy(self._model)
#
#     def _predict_f_internal(self, X: np.array) -> tuple[np.array, np.array]:
#         X_trans = self._pipe.transform(X)
#         y_pred, var = self._model.predict_f(X_trans)
#         y_pred, var = y_pred.numpy(), var.numpy()
#         if self._scaler_y is not None and self._rescale_y:
#             y_pred = self._scaler_y.inverse_transform(y_pred)
#             var = var * np.power(self._scaler_y.scale_[0], 2)
#         return y_pred.numpy(), var.numpy()
#
#     def _predict_y_internal(self, X: np.array) -> tuple[np.array, np.array]:
#         X_trans = self._pipe.transform(X)
#         y_pred, var = self._model.predict_y(X_trans)
#         y_pred, var = y_pred.numpy(), var.numpy()
#         if self._scaler_y is not None and self._rescale_y:
#             y_pred = self._scaler_y.inverse_transform(y_pred)
#             var = var * np.power(self._scaler_y.scale_[0], 2)
#         return y_pred.numpy(), var.numpy()
#
#     def predict_f(self, X: dict) -> tuple[np.array, np.array]:
#         X = self._unpack_func(X)
#         y_pred, var = self._predict_f_internal(X)
#         return y_pred, var
#
#     def predict_y(self, X: dict) -> tuple[np.array, np.array]:
#         X = self._unpack_func(X)
#         y_pred, var = self._predict_y_internal(X)
#         return y_pred, var
#
#
# class AdapterGPy(AbstractModelInterface):
#
#     def __init__(self, path_to_pkl: pl.Path, rescale_y: bool = True) -> None:
#         with open(path_to_pkl, "rb") as file:
#             pickle_obj = dill.load(file)
#         self._model = pickle_obj["model"]
#         self._unpack_func = pickle_obj["unpack_dict_func"]
#         self._scaler_X = pickle_obj["scaler_X"]
#         self._pca_X = pickle_obj["pca_X"]
#         self._scaler_y = pickle_obj["scaler_y"]
#         list_transform = []
#         if self._scaler_X is not None:
#             list_transform.append(("scaler_X", self._scaler_X))
#         if self._pca_X is not None:
#             list_transform.append(("pca_X", self._pca_X))
#         self._pipe = Pipeline(list_transform)
#         self._rescale_y = rescale_y
#
#     @property
#     def model(self):
#         return deepcopy(self._model)
#
#     def _predict_f_internal(self, X: np.array) -> tuple[np.array, np.array]:
#         X_trans = self._pipe.transform(X)
#         y_pred, var = self._model.predict_noiseless(X_trans)
#         if self._scaler_y is not None and self._rescale_y:
#             y_pred = self._scaler_y.inverse_transform(y_pred)
#             var = var * np.power(self._scaler_y.scale_[0], 2)
#         return y_pred, var
#
#     def _predict_y_internal(self, X: np.array) -> tuple[np.array, np.array]:
#         X_trans = self._pipe.transform(X)
#         y_pred, var = self._model.predict(X_trans)
#         if self._scaler_y is not None and self._rescale_y:
#             y_pred = self._scaler_y.inverse_transform(y_pred)
#             var = var * np.power(self._scaler_y.scale_[0], 2)
#         return y_pred, var
#
#     def predict_f(self, X: dict) -> tuple[np.array, np.array]:
#         X = self._unpack_func(X)
#         y_pred, var = self._predict_f_internal(X)
#         return y_pred, var
#
#     def predict_y(self, X: dict) -> tuple[np.array, np.array]:
#         X = self._unpack_func(X)
#         y_pred, var = self._predict_y_internal(X)
#         return y_pred, var
