import wandb as wb
from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.abstract_base_classes.experiment_tracker import AbstractExperimentTracker


class WandBTracker(AbstractExperimentTracker):

    def __init__(self, tracker_config: DictConfig, tracked_config: DictConfig):
        self._initial_tracker_config: DictConfig = tracker_config.copy()
        self._tracker_config: DictConfig = self._initial_tracker_config.copy()
        self._tracked_config: DictConfig = tracked_config.copy()
        self._run = None
        self._ready: bool = False

    @property
    def config(self) -> DictConfig:
        return self._tracker_config

    @config.setter
    def config(self, c):
        self._tracker_config = c

    def step(self, log_variables: dict[str, dict[str, float]], step_index: int) -> None:
        if self._ready:
            try:
                for category, variables in log_variables.items():
                    for name, value in variables.items():
                        self._run.log({f"{category}/{name}": value}, step_index)
            except Exception as e:
                self.close()
                raise e
        else:
            raise Exception("Cannot call step() before calling reset().")

    def reset(self, log_variables: dict[str, dict[str, float]], step_index: int) -> None:
        self.close()
        try:
            self._tracker_config = self._initial_tracker_config.copy()
            tracked_config_container = OmegaConf.to_container(self._tracked_config)
            self._run = wb.init(config=tracked_config_container, **self._tracker_config)
            self._ready = True
            self.step(log_variables, step_index)
        except Exception as e:
            self.close()
            raise e

    def close(self) -> None:
        if self._run:
            self._run.finish()
            self._tracker_config = OmegaConf.create()
            self._run = None
            self._ready = False
