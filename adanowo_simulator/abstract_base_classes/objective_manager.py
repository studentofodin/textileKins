from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractObjectiveManager(ABC):
    """Abstract class for an objective manager.

    An objective manager calculates an objective value depending on state and outputs.
    It is a necessary member of an
    :py:class:'~adanowo_simulator.abstract_base_classes.environment.AbstractEnvironment'."""

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of the objective manager."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c) -> None:
        pass

    @abstractmethod
    def step(self, state: dict[str, float], outputs: dict[str, float],
             setpoint_constraints_met: dict[str, bool], dependent_variable_constraints_met: dict[str, bool]) -> \
            tuple[float, dict[str, bool]]:
        """Checks output bound violations and calculates an objective value.

        Parameters
        -------
        state : dict[str, float]
        outputs : dict[str, float]
        setpoint_constraints_met : dict[str, bool]
            A dictionary which indicates for each setpoint bound if it was satisfied (True) or not (False) if the
            actions would be applied blindly. This means, one variable with an upper and a lower bound results in
            two key value pairs.
        dependent_variable_constraints_met : dict[str, bool]
            A dictionary which indicates for each dependent variable bound if it was satisfied (True) or not (False) if
            the actions would be applied blindly. This means, one variable with an upper and a lower bound results in
            two key value pairs.

        Returns
        -------
        float
            Objective value.
        dict[str, bool]
            A dictionary which indicates for each output bound (usually known from :py:attr:'config') if it is
            satisfied (True) or not (False). This means, one variable with an upper and a lower bound results in
            two key value pairs.
        """
        pass

    @abstractmethod
    def reset(self, initial_state: dict[str, float], initial_outputs: dict[str, float],
              setpoint_constraints_met_initially: dict[str, bool],
              dependent_variable_constraints_met_initially: dict[str, bool]) -> tuple[float, dict[str, bool]]:
        """Resets the objective manager to initial values and calculates an initial objective value.

        Parameters
        -------
        initial_state : dict[str, float]
        initial_outputs : dict[str, float]
        setpoint_constraints_met_initially : dict[str, bool]
            A dictionary which indicates for each setpoint bound if it is satisfied (True) or not (False) by the initial
            septoints. This means, one variable with an upper and a lower bound results in two key value pairs.
        dependent_variable_constraints_met_initially : dict[str, bool]
            A dictionary which indicates for each dependent variable bound if it is satisfied (True) or not (False) by
            the initial dependent variables. This means, one variable with an upper and a lower bound results in
            two key value pairs.

        Returns
        -------
        float
            Initial objective value.
        dict[str, bool]
            A dictionary which indicates for each setpoint bound (usually known from :py:attr:'config') if it is
            satisfied (True) or not (False) by the initial outputs. This means, one variable with an upper and a lower
            bound results in two key value pairs.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the objective manager."""
        pass
