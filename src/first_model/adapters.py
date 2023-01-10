import numpy as np
from sklearn.pipeline import Pipeline
from copy import deepcopy

from utils.interfaces import ModelInterface

# not needed, can be deleted if it leads to import errors
class AdapterSVGP(ModelInterface):
    from gpflow.models import SVGP
    """
    Adapts a GPFlow SVGP model to interface with the app.
    """

    def __init__(self, pickle_obj: dict, model_properties: dict, rescale_y: bool = True) -> None:
        self._model = pickle_obj["model"]
        self._unpack_func = pickle_obj["unpack_dict_func"]
        self._scaler_X = pickle_obj["scaler_X"]
        self._pca_X = pickle_obj["pca_X"]
        self._scaler_y = pickle_obj["scaler_y"]
        list_transform = []
        if self._scaler_X is not None:
            list_transform.append(("scaler_X", self._scaler_X))
        if self._pca_X is not None:
            list_transform.append(("pca_X", self._pca_X))
        self._pipe = Pipeline(list_transform)

        self._model_properties = model_properties
        self._rescale_y = rescale_y

    @property
    def model(self) -> SVGP:
        return deepcopy(self._model)

    @property
    def model_properties(self) -> dict:
        return deepcopy(self._model_properties)

    def predict_f_internal(self, X: np.array) -> [np.array, np.array]:
        X_trans = self._pipe.transform(X)
        y_pred, var = self._model.predict_f(X_trans)
        y_pred, var = y_pred.numpy(), var.numpy()
        if self._scaler_y is not None and self._rescale_y:
            y_pred = self._scaler_y.inverse_transform(y_pred)
            var = var * self._scaler_y.scale_[0]
        return y_pred.numpy(), var.numpy()

    def predict_y_internal(self, X: np.array) -> [np.array, np.array]:
        X_trans = self._pipe.transform(X)
        y_pred, var = self._model.predict_y(X_trans)
        y_pred, var = y_pred.numpy(), var.numpy()
        if self._scaler_y is not None and self._rescale_y:
            y_pred = self._scaler_y.inverse_transform(y_pred)
            var = var * self._scaler_y.scale_[0]
        return y_pred.numpy(), var.numpy()

    def predict_f(self, X: dict) -> [np.array, np.array]:
        X = self._unpack_func(X)
        y_pred, var = self.predict_f_internal(X)
        return y_pred, var

    def predict_y(self, X: dict) -> [np.array, np.array]:
        X = self._unpack_func(X)
        y_pred, var = self.predict_y_internal(X)
        return y_pred, var


class AdapterGPy(ModelInterface):
    from GPy.models.gp_heteroscedastic_regression import \
        GPHeteroscedasticRegression
    from GPy.models.gp_regression import GPRegression

    def __init__(self, pickle_obj: dict, model_properties: dict, rescale_y: bool = True) -> None:
        self._model = pickle_obj["model"]
        self._unpack_func = pickle_obj["unpack_dict_func"]
        self._scaler_X = pickle_obj["scaler_X"]
        self._pca_X = pickle_obj["pca_X"]
        self._scaler_y = pickle_obj["scaler_y"]
        list_transform = []
        if self._scaler_X is not None:
            list_transform.append(("scaler_X", self._scaler_X))
        if self._pca_X is not None:
            list_transform.append(("pca_X", self._pca_X))
        self._pipe = Pipeline(list_transform)

        self._model_properties = model_properties
        self._rescale_y = rescale_y

    @property
    def model(self) -> GPRegression:
        return deepcopy(self._model)

    @property
    def model_properties(self) -> dict:
        return deepcopy(self._model_properties)

    def predict_f_internal(self, X: np.array) -> [np.array, np.array]:
        X_trans = self._pipe.transform(X)
        y_pred, var = self._model.predict_noiseless(X_trans)
        if self._scaler_y is not None and self._rescale_y:
            y_pred = self._scaler_y.inverse_transform(y_pred)
            var = var * self._scaler_y.scale_[0]
        return y_pred, var

    def predict_y_internal(self, X: np.array) -> [np.array, np.array]:
        X_trans = self._pipe.transform(X)
        y_pred, var = self._model.predict(X_trans)
        if self._scaler_y is not None and self._rescale_y:
            y_pred = self._scaler_y.inverse_transform(y_pred)
            var = var * self._scaler_y.scale_[0]
        return y_pred, var

    def predict_f(self, X: dict) -> np.array:
        X = self._unpack_func(X)
        y_pred, var = self.predict_f_internal(X)
        return y_pred, var

    def predict_y(self, X: dict) -> np.array:
        X = self._unpack_func(X)
        y_pred, var = self.predict_y_internal(X)
        return y_pred, var