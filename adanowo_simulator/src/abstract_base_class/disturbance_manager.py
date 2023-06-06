from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractDisturbanceManager(ABC):
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
    def n_disturbances(self) -> int:
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
