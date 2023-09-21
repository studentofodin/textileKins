from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager


class DisturbanceManager(AbstractDisturbanceManager):
    def __init__(self, config: DictConfig):
        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = self._initial_config.copy()
        self._ready: bool = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self) -> dict[str, float]:
        if self._ready:
            disturbances = OmegaConf.to_container(self._config.disturbances)
            return disturbances
        else:
            raise Exception("Cannot call step() before calling reset().")

    def reset(self) -> dict[str, float]:
        self._config = self._initial_config.copy()
        self._ready = True
        disturbances = self.step()
        return disturbances

    def close(self) -> None:
        pass
