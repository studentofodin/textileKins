from abc import ABC, abstractmethod
from omegaconf import DictConfig


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
    def init_experiment(self):
        """
        initialize wandb.
        """
        pass

    @abstractmethod
    def log(self, logVariables: dict[str, float]) -> None:
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
