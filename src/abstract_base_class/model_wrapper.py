from abc import ABC, abstractmethod
import numpy as np
from typing import List
from typing import Tuple
from typing import Dict
from omegaconf import DictConfig

from src.abstract_base_class.model_interface import AbstractModelInterface

class AbstractModelWrapper(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @property
    @abstractmethod
    def n_models(self) -> int:
        pass

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def model_names(self) -> Dict[str, str]:
        pass

    @property
    @abstractmethod
    def model_props(self) -> Dict[str, any]:
        pass

    @property
    @abstractmethod
    def machine_models(self) -> Dict[str, AbstractModelInterface]:
        pass

    @property
    @abstractmethod
    def means(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def vars(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def outputs(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def outputs_array(self) -> np.array:
        pass

    @abstractmethod
    def call_models(self, input: Dict[str, float]) -> None:
        """
        call all machine models to determine mean and variance (predict_y() or predict_f()) from input and assign
        values to properties means and vars.
        """
        pass

    @abstractmethod
    def interpret_model_outputs(self) -> None:
        """
        sample output distributions dependent on corresponding mean and variance and assign values to properties
        outputs and outputs_array.
        """
        pass

    @abstractmethod
    def get_outputs(self, input: Dict[str, float]) -> Tuple[np.array, dict]:
        """
        call methods call_models() and interpret_model_outputs() and return properties outputs and outputs_array.
        """
        pass