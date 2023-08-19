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

    @abstractmethod
    def step(self) -> dict[str, float]:
        pass

    @abstractmethod
    def reset(self) -> dict[str, float]:
        pass
