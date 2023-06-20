from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.abstract_base_class.disturbance_manager import AbstractDisturbanceManager

class DisturbanceManager(AbstractDisturbanceManager):

    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._config = config.copy()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self) -> dict[str, float]:
        disturbances = OmegaConf.to_container(self._config.disturbances)
        return disturbances

    def reset(self) -> dict[str, float]:
        self._config = self._initial_config.copy()
        disturbances = OmegaConf.to_container(self._config.disturbances)
        return disturbances
