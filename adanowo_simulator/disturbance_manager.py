from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager


class DisturbanceManager(AbstractDisturbanceManager):

    def __init__(self, initial_disturbances: DictConfig):
        self._initial_disturbances = initial_disturbances.copy()
        self._disturbances = None
        self._ready = False

    @property
    def disturbances(self) -> DictConfig:
        return self._disturbances

    @disturbances.setter
    def disturbances(self, c):
        self._disturbances = c

    def step(self) -> dict[str, float]:
        if self._ready:
            disturbances = OmegaConf.to_container(self._disturbances)
            return disturbances
        else:
            raise Exception("Cannot call step() before calling reset().")


    def reset(self) -> dict[str, float]:
        self._disturbances = self._initial_disturbances.copy()
        disturbances = OmegaConf.to_container(self._disturbances)
        self._ready = True
        return disturbances
