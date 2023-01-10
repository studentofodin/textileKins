from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from typing import List

from src.abstract_base_class.model_interface import AbstractModelInterface
from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper



class AbstractModelWrapper(ABC):

    @staticmethod
    @abstractmethod
    def load_model(model_props: dict) -> AbstractModelInterface:
        pass

    @property
    @abstractmethod
    def machine_models(self) -> List[AbstractModelInterface, ...]:
        pass

    @abstractmethod
    def interpret_model_outputs(self, mean: np.array, var: np.array) -> np.array:
        pass

    @abstractmethod
    def call_models(self, input: dict, latent: bool) -> Tuple[np.array, np.array]:
        pass

    @abstractmethod
    def call_models_internal(self, input: np.array, latent: bool) -> Tuple[np.array, np.array]:
        pass

    @abstractmethod
    def get_output(self, input: dict, latent: bool) -> np.array:
        pass

    @abstractmethod
    def get_output_internal(self, input: np.array, latent: bool) -> np.array:
        pass
