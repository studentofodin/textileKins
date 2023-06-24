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
        """
        log the given log_variables in wandb.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        finish wandb run and reset to initial values.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        finish experiment tracking run.
        """
        pass
