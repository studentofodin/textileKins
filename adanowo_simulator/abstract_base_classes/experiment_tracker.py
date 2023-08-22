from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Any


class AbstractExperimentTracker(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, log_variables: dict[str, Any], step_index: int) -> None:
        pass

    @abstractmethod
    def reset(self, log_variables: dict[str, Any], step_index: int) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
