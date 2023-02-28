from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.model_interface import AbstractModelInterface


class AbstractModelWrapper(ABC):

    @property
    @abstractmethod
    def n_models(self) -> int:
        pass

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def model_names(self) -> dict[str, str]:
        pass

    @property
    @abstractmethod
    def model_props(self) -> dict[str, any]:
        pass

    @property
    @abstractmethod
    def machine_models(self) -> dict[str, AbstractModelInterface]:
        pass

    @property
    @abstractmethod
    def means(self) -> dict[str, float]:
        pass

    @property
    @abstractmethod
    def vars(self) -> dict[str, float]:
        pass

    @property
    @abstractmethod
    def outputs(self) -> dict[str, float]:
        pass

    @property
    @abstractmethod
    def outputs_array(self) -> np.array:
        pass

    @abstractmethod
    def _call_models(self, input_model: dict[str, float], latent) -> None:
        """
        call all machine models to determine mean and variance (predict_y() or predict_f()) from input and assign
        values to properties means and vars.
        """
        pass

    @abstractmethod
    def _interpret_model_outputs(self) -> None:
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
