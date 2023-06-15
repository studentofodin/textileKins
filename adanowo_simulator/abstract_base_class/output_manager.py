from abc import ABC, abstractmethod
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

    @abstractmethod
    def step(self, controls: dict[str, float], disturbances: dict[str, float]) -> dict[str, float]:
        """
        get sampled output values from inputs for each model.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        reset to initial values.
        """
        pass
