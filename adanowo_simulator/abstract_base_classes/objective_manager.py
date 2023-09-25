from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractObjectiveManager(ABC):
    """Abstract class for an objective manager.

    An objective manager calculates an objective value depending on state and outputs.
    It is not useful on its own and should be a member of an
    :py:class:'~adanowo_simulator.abstract_base_classes.environment.AbstractEnvironment'."""

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of an output manager. Must contain the output bounds."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c) -> None:
        pass

    @abstractmethod
    def step(self, state: dict[str, float], outputs: dict[str, float],
             setpoint_constraints_met: dict[str, bool], dependent_variable_constraints_met: dict[str, bool]) -> \
            tuple[float, dict[str, bool]]:
        """Checks output bound exceedances and calculates an objective value.

        Parameters
        -------
        state : dict[str, float]
        outputs : dict[str, float]
        setpoint_constraints_met : dict[str, bool]
            A dictionary which contains for each setpoint bound if it was exceeded (False) or not (True)
            if the actions would be applied blindly.
            This means meeting the constraints corresponds to the value True and violating them corresponds to the value False.
        dependent_variable_constraints_met : dict[str, bool]
            A dictionary which contains for each dependent variable bound if it was exceeded (False) or not (True)
            if the actions would be applied blindly.
            This means meeting the constraints corresponds to the value True and violating them corresponds to the value False.

        Returns
        -------
        float
            Objective value.
        dict[str, bool]
            A dictionary which contains for each output bound if it is exceeded (False) or not (True).
            This means meeting the constraints corresponds to the value True and violating them corresponds to the value False.
            The objective value may depend on this dictionary.
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
            A dictionary which contains for each setpoint bound if it is exceeded (False) or not (True) by the initial setpoints.
            This means being within bounds corresponds to the value True and being out of bounds to the value False.
        dependent_variable_constraints_met_initially : dict[str, bool]
            A dictionary which contains for each dependent variable bound if it is exceeded (False) or not (True) by the initial dependent variables.
            This means being within bounds corresponds to the value True and being out of bounds to the value False.

        Returns
        -------
        float
            Initial objective value.
        dict[str, bool]
            A dictionary which contains for each output bound if it is exceeded (False) or not (True) by the initial outputs.
            This means meeting the constraints corresponds to the value True and violating them corresponds to the value False.
            The objective value may depend on this dictionary.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the objective manager."""
        pass
