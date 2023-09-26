from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractDisturbanceManager(ABC):
    """Abstract class for a disturbance manager.

    A disturbance manager gets disturbances.
    It is not useful on its own and should be a member of an
    :py:class:'~adanowo_simulator.abstract_base_classes.environment.AbstractEnvironment'."""
    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of the disturbance manager."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self) -> dict[str, float]:
        """Gets disturbances and returns them."""
        pass

    @abstractmethod
    def reset(self) -> dict[str, float]:
        """Resets the disturbance manager to initial values and returns the initial disturbances."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the disturbance manager."""
        pass
