from abc import ABC, abstractmethod
import numpy as np
from omegaconf import DictConfig


class AbstractOutputManager(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        pass

    @abstractmethod
    def step(self, controls: dict[str, float], disturbances: dict[str, float]) -> dict[str, float]:
        """
        get sampled output values from inputs for each model.
        """
        pass

    @abstractmethod
    def update(self, changed_outputs: list[str]) -> None:
        """
        update the models for the outputs listed in changed_outputs based on the entries in the own config.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        reset to initial values.
        """
        pass

    @abstractmethod
    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        """
        load model corresponding to model_name.
        allocate this model to the output output_name.
        """
        pass
