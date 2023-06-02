import wandb as wb
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class ExperimentTracker(AbstractExperimentTracker):

    def __init__(self, logger_config: DictConfig, logged_config: DictConfig):
        self._initial_config = logger_config.copy()
        self._logged_config = logged_config.copy()
        self._logger_config = logger_config.copy()
        self._run = None

    @property
    def config(self) -> DictConfig:
        return self._logger_config

    @config.setter
    def config(self, c):
        self._logger_config = c

    def init_run(self):
        logged_config_container = OmegaConf.to_container(self._logged_config)
        self._run = wb.init(config=logged_config_container, **self._logger_config)

    def log(self, log_variables: dict) -> None:
        self._run.log(log_variables)

    def reset(self) -> None:
        if self._run:
            self._run.finish()
        self._run = None
        self._logger_config = self._initial_config.copy()
