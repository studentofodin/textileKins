import wandb as wb
from omegaconf import DictConfig

from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class ExperimentTracker(AbstractExperimentTracker):

    def __init__(self, config: DictConfig):
        self._config = config
        wb.init(**self._config)

    @property
    def config(self) -> DictConfig:
        return self._config

    def initWandB(self) -> None:
        wb.init(**self._config)

    def log(self, logVariables: dict) -> None:
        wb.log(logVariables)

