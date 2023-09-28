from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Any


class AbstractExperimentTracker(ABC):
    """Abstract class for an experiment tracker.

    An experiment tracker tracks a sequence of steps of an environment, e.g. by saving some variables in files or
    creating diagrams. Thus, it is not useful on its own and should be a member of an
    :py:class:'~adanowo_simulator.abstract_base_classes.environment.AbstractEnvironment'.
    """

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of the experiment tracker."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, log_variables: dict[str, dict[str, float]], step_index: int) -> None:
        """Performs a variable log.

        Parameters
        -------
        log_variables : dict[str, dict[str, float]]
            Variables to log. The first key indicates a category/group of variables, the second one the name of one variable.
        step_index : int
            Index of the current step.
        """
        pass

    @abstractmethod
    def reset(self, initial_log_variables: dict[str, dict[str, float]]) -> None:
        """Resets the experiment tracker to initial values and logs passed initial variables."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the experiment tracker."""
        pass
