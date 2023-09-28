from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractOutputManager(ABC):
    """Abstract class for an output manager.

    An output manager gets outputs, either via measurments or via models.
    It is not useful on its own and should be a member of an
    :py:class:'~adanowo_simulator.abstract_base_classes.environment.AbstractEnvironment'.
    """

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of the output manager."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, state: dict[str, float]) -> dict[str, float]:
        """Gets outputs (via measurement or models) and returns them."""
        pass

    @abstractmethod
    def reset(self, initial_state: dict[str, float]) -> dict[str, float]:
        """Resets the output manager to initial values and returns initial outputs."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the output manager."""
        pass

