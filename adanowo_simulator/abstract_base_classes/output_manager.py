from abc import ABC, abstractmethod

from omegaconf import DictConfig
import numpy as np


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
    def step(self, inputs: dict[str, float]) -> dict[str, float]:
        pass

    @abstractmethod
    def reset(self, state: dict[str, float]) -> dict[str, np.array]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

