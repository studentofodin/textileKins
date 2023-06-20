from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager


class DisturbanceManager(AbstractDisturbanceManager):

    def __init__(self, initial_disturbances: DictConfig):
        self._initial_disturbances = initial_disturbances.copy()
        self._n_disturbances = len(initial_disturbances)
        self._disturbances = self._initial_disturbances.copy()

    @property
    def disturbances(self) -> DictConfig:
        return self._disturbances

    @disturbances.setter
    def disturbances(self, c):
        self._disturbances = c

    def step(self) -> dict[str, float]:
        disturbances = OmegaConf.to_container(self._disturbances)
        return disturbances

    def reset(self) -> dict[str, float]:
        self._disturbances = self._initial_disturbances.copy()
        disturbances = OmegaConf.to_container(self._disturbances)
        return disturbances
