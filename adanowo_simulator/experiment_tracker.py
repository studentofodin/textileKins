import wandb as wb
from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.abstract_base_class.experiment_tracker import AbstractExperimentTracker


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

    def init_experiment(self):
        exp_config = OmegaConf.to_container(self._experiment_config)
        self._run = wb.init(config=exp_config, **self._config)

    def step(self, log_variables: dict[str, dict[str, float]], step_index: int) -> None:
        for category, variables in log_variables.items():
            for name, value in variables.items():
                self._run.log({f"{category}/{name}": value}, step_index)

    def reset(self) -> None:
        if self._run:
            self._run.finish()
        self._run = None
        self._logger_config = self._initial_config.copy()
