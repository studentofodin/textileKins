from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractExperimentTracker(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @abstractmethod
    def initWandB(self) -> None:
        """
        initialise wandb.
        """
        pass

    @abstractmethod
    def log(self, logVariables: dict) -> None:
        """
        log logVariables in wandb.
        """
        pass