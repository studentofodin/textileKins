from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractDisturbanceManager(ABC):
    @property
    @abstractmethod
    def disturbances(self) -> DictConfig:
        pass

    @disturbances.setter
    @abstractmethod
    def disturbances(self, c):
        pass

    @abstractmethod
    def step(self) -> dict[str, float]:
        """
        get disturbances.
        """
        pass

    @abstractmethod
    def reset(self) -> dict[str, float]:
        """
        reset to initial values.
        """
