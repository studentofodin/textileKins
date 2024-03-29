from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.objective_manager import AbstractObjectiveManager
from adanowo_simulator.abstract_base_classes.action_manager import AbstractActionManager
from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager
from adanowo_simulator.abstract_base_classes.experiment_tracker import AbstractExperimentTracker
from adanowo_simulator.abstract_base_classes.scenario_manager import AbstractScenarioManager


class AbstractEnvironment(ABC):
    """Abstract class for an environment that represent the stationary behaviour of a nonwoven production process.

    From an outside point of view an environment receives *actions* from an *agent*,
    processes them and returns *observations* as well as an *objective value*.
    This is called a *step*.

    Internally, process variables are used.
    As written above only stationary behaviour is considered.
    This means the process variables are considered after all transient phenomena have decayed.
    The process variables are:
    * *Disturbances* - Variables that can not be manipulated.
    Some of them might be manipulable in general but the considered configuration predefines their values.
    * *Setpoints* - Variables that can be manipulated with the purpose of maximizing the objective value.
    They are calculated from the actions either in an absolute or relative manner.
    | Absolute manner: setpoints(t) = actions(t)
    | Relative manner: setpoints(t) = setpoints(t-1) + actions(t)
    The absolute manner results in a static system, whereas the relative manner results in a dynamic system.
    * *Dependent variables* - Important variables that are calculated from the setpoints and disturbances, i.e.
    dependent_variables(t) = f(setpoints(t), disturbances(t)).
    They can be introduced either because their bounds have to be checked or
    their values are needed at other places in the simulation.
    * *State* - Combination of disturbances, setpoints and dependent variables.
    * *Outputs* - Variables that can not be calculated directly.
    They depend not only on the state but on many more variables that are not incorporated in the state.
    Furthermore, the relationship between the state and the outputs can be complex.
    Thus, on a physical machine, outputs have to be measured.
    In a simulation environment, output models have to be used that approximate real-world behavior.

    Setpoints, dependent variables and outputs may be bounded. If so, their bounds are checked.
    An environment's behaviour in the case of bound violations depends on the specific implementation.

    The observations and the objective value of an environment also depend on the specific implementation.
    There is no restriction regarding the observations, they can consist of anything that may be useful for the agent.
    The objective value is calculated from state and outputs, i.e. objective_value(t) = h(state(t), outputs(t)) and
    often incorporates bound violations of these variables.

    .. note::
    Assume variable y depends on a variable x.
    * We use the term *calculation* if, given x, a value for y is directly returned.
    * We use the term *model* if, given x, a probability distribution for y is returned.
    To get an output value, you have to sample from the modeled distribution.

    An environment consists of several members, each responsible for assigned subtasks.
    An environment as well as its members each have the attribute :py:attr:'config'
    which allows the user to configure them.
    We recommend *Hydra* (https://hydra.cc/) for configuration management.
    A scenario describes changes in the :py:attr:'config' of the members during a sequence of steps.
    The members are:
    * :py:attr:'disturbance_manager' - Gets the disturbances and returns them.
    * :py:attr:'action_manager' - Checks if the setpoint and dependent variable bounds would be satisfied
    if the actions received from the agent were applied blindly and returns the results of the checks.
    Based on the check results, calculates the setpoints and dependent variables and returns them.
    * :py:attr:'output_manager' - Gets the outputs (via measurement or model)  and returns them.
    * :py:attr:'objective_manager' - Checks if the output bounds are satisfied and returns the results of the checks.
    Calculates the objective value (which may depend on the results of the output bound checks) and returns it.
    * :py:attr:'scenario_manager' - Implements a scenario by changing the :py:attr:'config' of the other members.
    * :py:attr:'experiment_tracker' - Tracks a sequence of steps,
    e.g. by saving some variables in files or creating diagrams.

    The API methods of an environment are:
    * :py:meth:step - Updates the environment with actions returning observations and an objective value.
    * :py:meth:reset - Resets the environment to initial process variable values returning initial observations and objective value.
    Required before a sequence of steps.
    * :py:meth:close - Closes the environment. Important if the environment starts background processes that have to be shutdown.
    Thus, it is recommended to always call it if the environment is not needed anymore.

    .. note::
    The API of an environment mainly works with dictionaries, more specifically dict[str, float], whose keys represent
    names of numerical variables.
    """

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of the environment."""
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @property
    @abstractmethod
    def disturbance_manager(self) -> AbstractDisturbanceManager:
        """Gets the disturbances and returns them. Readonly."""
        pass

    @property
    @abstractmethod
    def action_manager(self) -> AbstractActionManager:
        """Processes the actions and disturbances into setpoints and dependent variables. Readonly.

        Checks if the setpoint and dependent variable bounds would be satisfied if the actions received from the agent
        were applied blindly and returns the results of the checks. Based on the check results, calculates the setpoints
        and dependent variables and returns them.
        """
        pass

    @property
    @abstractmethod
    def output_manager(self) -> AbstractOutputManager:
        """Gets the outputs (via measurement or model). Readonly."""
        pass

    @property
    @abstractmethod
    def objective_manager(self) -> AbstractObjectiveManager:
        """Calculates the objective value. Readonly.

        Checks if the output bounds are satisfied and returns the results of the checks. Calculates the objective value
        (which may depend on the results of the output bound checks) and returns it.
        """
        pass

    @property
    @abstractmethod
    def scenario_manager(self) -> AbstractScenarioManager:
        """Implements a scenario by changing the :py:attr:'config' of the other members. Readonly."""
        pass

    @property
    @abstractmethod
    def experiment_tracker(self) -> AbstractExperimentTracker:
        """Tracks a sequence of steps, e.g. by saving some variables in files or creating diagrams. Readonly."""
        pass

    @property
    @abstractmethod
    def step_index(self):
        """Tracks the number of steps. Readonly."""
        pass

    @abstractmethod
    def step(self, actions: dict) -> tuple[float, dict[str, float], dict[str, float], DictConfig]:
        """Updates the environment with actions returning an objective value,
        the current state as well as process outputs and quality bounds.
        """
        pass

    @abstractmethod
    def reset(self) -> tuple[float, dict[str, float], dict[str, float], DictConfig]:
        """Resets the environment to initial process variable values returning an objective value,
        the current state as well as process outputs and quality bounds. Required before a sequence of steps.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the environment.

        Important if the environment starts background processes that have to be shutdown.
        Thus, it is recommended to always call it if the environment is not needed anymore.
        """
        pass
