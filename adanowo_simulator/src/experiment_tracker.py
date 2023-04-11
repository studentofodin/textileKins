import wandb as wb
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class ExperimentTracker(AbstractExperimentTracker):

    def __init__(self, config: DictConfig, experiment_config: DictConfig):
        self._initialconfig = config.copy()
        self._experiment_config = experiment_config.copy()
        self._run = None
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def initRun(self):
        exp_config = OmegaConf.to_container(self._experiment_config)
        self._run = wb.init(config=exp_config, **self._config)

    def log(self, logVariables: dict) -> None:
        self._run.log(logVariables)

    def reset(self) -> None:
        if self._run:
            self._run.finish()
        self._run = None
        self._config = self._initialconfig.copy()



