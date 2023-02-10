from abc import ABC, abstractmethod
import numpy as np
from typing import List
from typing import Tuple

from src.abstract_base_class.model_interface import AbstractModelInterface

class AbstractModelWrapper(ABC):

    @property
    @abstractmethod
    def n_models(self) -> int:
        pass

    @property
    @abstractmethod
    def model_names(self) -> dict:
        pass

    @property
    @abstractmethod
    def model_props(self) -> dict:
        pass

    @property
    @abstractmethod
    def means(self) -> dict:
        pass

    @property
    @abstractmethod
    def vars(self) -> dict:
        pass

    @property
    @abstractmethod
    def outputs(self) -> dict:
        pass

    @property
    @abstractmethod
    def outputs_array(self) -> np.array:
        pass

    @property
    @abstractmethod
    def machine_models(self) -> dict:
        pass

    @property
    @abstractmethod
    def output_names(self) -> List:
        pass

    @abstractmethod
    def call_models(self, input: dict, latent: bool) -> None:
        pass

    @abstractmethod
    def interpret_model_outputs(self) -> dict:
        pass

    @abstractmethod
    def get_outputs(self, input: dict, latent: bool) -> Tuple[np.array, dict]:
        pass