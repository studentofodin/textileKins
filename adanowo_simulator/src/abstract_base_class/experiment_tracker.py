from abc import ABC, abstractmethod

from omegaconf import DictConfig


class AbstractExperimentTracker(ABC):

    @abstractmethod
    def initRun(self, exp_config: DictConfig) -> None:
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
