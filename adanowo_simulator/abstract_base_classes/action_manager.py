from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractActionManager(ABC):
    """Abstract class for an action manager.

    An action manager processes actions into setpoints and dependent variables.
    It is a necessary member of an
    :py:class:'~adanowo_simulator.abstract_base_classes.environment.AbstractEnvironment'.
    """

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of the action manager. Must contain the setpoint and dependent variable bounds."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, actions: dict[str, float], disturbances: dict[str, float]) -> \
            tuple[dict[str, float], dict[str, bool]]:
        """Calculates setpoints and dependent variables and checks boundary constraint violations of them.

        Checks if the setpoint and dependent variable bounds (usually known from :py:attr:'config') would be violated
        if the received actions were applied blindly.
        Based on the check results, calculates setpoints and dependent variables.

        If the setpoint and dependent variable bounds are satisfied when applying the actions, we have
        setpoints(t) = actions(t) if the action manager is implemented in an absolute manner and
        setpoints(t) = setpoints(t-1) + actions(t) if the action manager is implemented in a relative manner.
        The absolute manner results in a static system, whereas the relative manner results in a dynamic system.
        For the dependent variables we have an arbitrary, implementation dependent calculation
        dependent_variables(t) = f(setpoints(t), disturbances(t)).

        If one of the setpoint or dependent variable bounds would be violated if the actions were applied blindly,
        the resulting values depend on the specific implementation of an action manager.

        Returns
        -------
        dict[str, float]
            Setpoints.
        dict[str, float]
            Dependent variables.
        dict[str, bool]
            A dictionary which indicates for each setpoint bound if it was satisfied (True) or not (False) if the
            actions would be applied blindly. This means, one variable with an upper and a lower bound results in
            two key value pairs.
        dict[str, bool]
            A dictionary which indicates for each dependent variable bound if it was satisfied (True) or not (False) if
            the actions would be applied blindly. This means, one variable with an upper and a lower bound results in
            two key value pairs.
        """
        pass

    @abstractmethod
    def reset(self, initial_disturbances: dict[str, float]) -> \
            tuple[dict[str, float], dict[str, float], dict[str, bool], dict[str, bool]]:
        """Resets the action manager to initial values and calculates initial dependent variables.

        For the initial dependent variables we have an arbitrary calculation
        initial_dependent_variables = g(initial_setpoints, initial_disturbances).

        Returns
        -------
        dict[str, float]
            Initial setpoints.
        dict[str, float]
            Initial dependent variables.
        dict[str, bool]
            A dictionary which indicates for each setpoint bound if it is satisfied (True) or not (False) by the initial
            septoints. This means, one variable with an upper and a lower bound results in two key value pairs.
        dict[str, bool]
            A dictionary which indicates for each dependent variable bound if it is satisfied (True) or not (False) by
            the initial dependent variables. This means, one variable with an upper and a lower bound results in
            two key value pairs.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the action manager."""
        pass
