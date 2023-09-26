from abc import ABC, abstractmethod
from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager
from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.objective_manager import AbstractObjectiveManager


class AbstractScenarioManager(ABC):
    """Abstract class for a scenario manager.

    A scenario manager implements a scenario by changing the :py:attr:'config' of the members of an environment.
    Thus, it is not useful on its own and should itself be a member of an
    :py:class:'~adanowo_simulator.abstract_base_classes.environment.AbstractEnvironment'."""
    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of a scenario manager."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, step_index: int, disturbance_manager: AbstractDisturbanceManager,
             output_manager: AbstractOutputManager, objective_manager: AbstractObjectiveManager) -> None:
        """Changes the :py:attr:'config' of the passed environment members.
        Parameter 'step_index' may be important to determine what to change."""
        pass

    def reset(self) -> None:
        """Resets the scenario manager to initial values."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the scenario manager."""
        pass
