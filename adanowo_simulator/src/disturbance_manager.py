from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.abstract_base_class.disturbance_manager import AbstractDisturbanceManager

class DisturbanceManager(AbstractDisturbanceManager):

    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._n_disturbances = len(config.disturbances)
        self._config = None
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def n_disturbances(self) -> int:
        return self._n_disturbances

    def get_disturbances(self) -> dict[str, float]:
        disturbances = dict(self._config.disturbances)
        return disturbances

    def reset(self) -> dict[str, float]:
        self._config = self._initial_config.copy()
        disturbances = dict(self._config.disturbances)
        return disturbances
