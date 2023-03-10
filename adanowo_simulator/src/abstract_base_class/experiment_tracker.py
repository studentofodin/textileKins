from abc import ABC, abstractmethod

from omegaconf import DictConfig


class AbstractExperimentTracker(ABC):

    @abstractmethod
    def initTracker(self, exp_config: DictConfig) -> None:
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

    @abstractmethod
    def finish_experiment(self) -> None:
        """
        Tells the tracker to finish the experiment
        """
