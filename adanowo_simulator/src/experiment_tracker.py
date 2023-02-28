import wandb as wb
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class ExperimentTracker(AbstractExperimentTracker):

    def __init__(self, wb_config: DictConfig):
        self._wb_config = wb_config
        self._run = None

    def initTracker(self, exp_config: DictConfig):
        exp_config = OmegaConf.to_container(exp_config)
        self._run = wb.init(config=exp_config, **self._wb_config)

    def log(self, logVariables: dict) -> None:
        self._run.log(logVariables)

    def finish_experiment(self) -> None:
        self._run.finish()
        self._run = None
