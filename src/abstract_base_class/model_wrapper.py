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
    def machine_models(self) -> List[AbstractModelInterface]:
        pass

    @property
    @abstractmethod
    def n_models(self) -> int:
        pass

    @property
    @abstractmethod
    def means(self) -> np.array:
        pass

    @property
    @abstractmethod
    def vars(self) -> np.array:
        pass

    @property
    @abstractmethod
    def outputs(self) -> np.array:
        pass

    @abstractmethod
    def call_models(self, input: dict, latent: bool) -> None:
        pass

    @abstractmethod
    def interpret_model_outputs(self) -> None:
        pass

    @abstractmethod
    def get_outputs(self, input: dict, latent: bool) -> np.array:
        pass