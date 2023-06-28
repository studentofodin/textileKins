import wandb as wb
from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.abstract_base_classes.experiment_tracker import AbstractExperimentTracker


class WandBTracker(AbstractExperimentTracker):

    def __init__(self, tracker_config: DictConfig, tracked_config: DictConfig):
        self._initial_config = tracker_config.copy()
        self._tracked_config = tracked_config.copy()
        self._tracker_config = tracker_config.copy()
        self._run = None

    @property
    def config(self) -> DictConfig:
        return self._tracker_config

    @config.setter
    def config(self, c):
        self._tracker_config = c

    def init_experiment(self):
        tracked_config_container = OmegaConf.to_container(self._tracked_config)
        self._run = wb.init(config=tracked_config_container, **self._tracker_config)

    def step(self, log_variables: dict[str, dict[str, float]], step_index: int) -> None:
        for category, variables in log_variables.items():
            for name, value in variables.items():
                self._run.log({f"{category}/{name}": value}, step_index)

    def reset(self) -> None:
        self.shutdown()
        self._tracker_config = self._initial_config.copy()
        self.init_experiment()

    def shutdown(self) -> None:
        if self._run:
            self._run.finish()
        self._run = None
