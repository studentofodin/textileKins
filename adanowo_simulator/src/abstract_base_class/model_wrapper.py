from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.model_interface import AbstractModelInterface


class AbstractModelWrapper(ABC):

    @abstractmethod
    def _call_models(self, input_model: dict[str, float], latent) -> (dict[str, float], dict[str, float]):
        """
        call all machine models to determine mean and variance (predict_y() or predict_f()) from input and assign
        values to properties means and vars.
        """
        pass

    @abstractmethod
    def _interpret_model_outputs(self, mean_pred: dict[str, float], var_pred: dict[str, float]) \
            -> (np.array, dict[str, float]):
        """
        sample output distributions dependent on corresponding mean and variance and assign values to properties
        outputs and outputs_array.
        """
        pass

    @abstractmethod
    def get_outputs(self, input_model: dict[str, float]) -> tuple[np.array, dict]:
        """
        call methods call_models() and interpret_model_outputs() and return properties outputs and outputs_array.
        """
        pass
