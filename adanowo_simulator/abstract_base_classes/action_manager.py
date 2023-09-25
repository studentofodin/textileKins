from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractActionManager(ABC):
    """Abstract class for an action manager.

    An action manager processes actions into controls and dependent variables.
    It is not useful on its own and should be a member of an
    :py:class:'~adanowo_simulator.abstract_base_classes.environment.AbstractEnvironment'.
    """

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of an action manager. Must contain the control and dependent variable bounds."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, actions: dict[str, float], disturbances: dict[str, float]) -> \
            tuple[dict[str, float], dict[str, bool]]:
        """Calculates controls and dependent variables.

        Checks if the control and dependent variable bounds (known from :py:attr:'config') would be exceeded
        if the received actions were applied blindly.
        Based on the check results, calculates controls and dependent variables.

        If the control bounds are not exceeded we have
        controls(t) = actions(t) if the action manager is implemented in an absolute manner and
        controls(t) = controls(t-1) + actions(t) if the action manager is implemented in a relative manner.
        The absolute manner results in a static system, whereas the relative manner results in a dynamic system.
        For the dependent variables we have an arbitrary calculation
        dependent_variables(t) = g(controls(t), disturbances(t)).

        Returns
        -------
        dict[str, float]
            controls.
        dict[str, float]
            Dependent variables.
        dict[str, bool]
            A dictionary which contains for each control bound if it was exceeded (False) or not (True)
            if the actions would be applied blindly.
            This means meeting the constraints corresponds to the value True and violating them corresponds to the value False.
        dict[str, bool]
            A dictionary which contains for each dependent variable bound if it was exceeded (False) or not (True)
            if the actions would be applied blindly.
            This means meeting the constraints corresponds to the value True and violating them corresponds to the value False.
        """
        pass

    @abstractmethod
    def reset(self, initial_disturbances: dict[str, float]) -> \
            tuple[dict[str, float], dict[str, float], dict[str, bool], dict[str, bool]]:
        """Resets the action manager to initial values and calculates initial dependent variables from them.

        For the initial dependent variables we have an arbitrary calculation
        initial_dependent_variables = g(initial_controls, initial_disturbances).

        Returns
        -------
        dict[str, float]
            Initial controls.
        dict[str, float]
            Initial dependent variables.
        dict[str, bool]
            A dictionary which contains for each control bound if it is exceeded (False) or not (True) by the initial controls.
            This means being within bounds corresponds to the value True and being out of bounds to the value False.
        dict[str, bool]
            A dictionary which contains for each dependent variable bound if it is exceeded (False) or not (True) by the initial dependent variables.
            This means being within bounds corresponds to the value True and being out of bounds to the value False.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the action manager."""
        pass
