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
    """Abstract class for an environment that represent the stationary behaviour of a non-woven production process.

        From an outside point of view an environment receives *actions* from an *agent*,
        processes them and returns *observations* as well as a *reward*.
        This is called a *step*.

        Internally, process variables are used.
        As written above stationary behaviour is considered.
        This means the process variables are considered after all transient phenomena have decayed.
        The additional process variables are:
        * *Disturbances* - Variables that can not be manipulated.
        Some of them might be manipulatable in general but the considered configuration predefines their values.
        * *Controls* - Variables that can be manipulated with the purpose of maximizing the reward.
        They can be calculated from the actions either in an absolute or relative manner.
        | Absolute manner: controls(t) = actions(t)
        | Relative manner: controls(t) = controls(t-1) + actions(t)
        The absolute manner results in a static system, whereas the relative manner results in a dynamic system.
        * *Dependent variables* - Important variables that can be calculated from the controls and disturbances, i.e.
        dependent_variables(t) = g(controls(t), disturbances(t)).
        They are introduced either because their bounds have to be checked or
        their values are needed several at several points and by saving them their calculation is done only once instead of several times.
        * *State* - Combination of disturbances, controls and dependent variables.
        * *Outputs* - Variables that can not be calculated directly.
        They depend not only on the state but on many more variables that are not incorporated in the state.
        Furthermore, the relationship between the state and the outputs can be complex.
        Thus, outputs have to be measured on a physical machine.
        In a simulation environment models have to be used that are not capable to fully describe the real world.

        Controls, dependent variables and outputs may be bounded.
        If so, their bounds have to be checked.
        An environment's behaviour in the case of bound exceedance depends on the specific implementation.

        The observations of an environment also depend on the specific implementation.
        They have to be calculable from the process variables und usually are a subset of them.

        .. note::
        Assume variable y depends on a variable x.
        We use the term *calculation* if, given x, a value for y is directly returned.
        We use the term *model* if, given x,
        a probability distribution for y is returned. To get a value you have to sample from the modeled distribution.

        An environment consists of several members, each responsible for assigned subtasks.
        An environment as well as its members each have the attribute :py:attr:'config' which allows the user to configure them.
        We recommend *Hydra* (https://hydra.cc/) for configuration management.
        A scenario describes changes in the :py:attr:'config' of the members during a sequence of steps.
        The members are:
        * :py:attr:'disturbance_manager' - Gets the disturbances and returns them.
        * :py:attr:'action_manager' - Checks if the control and dependent variable bounds would be exceeded
        if the actions received from the agent were applied blindly and returns the results of the checks.
        Based on the check results, calculates the controls and dependent variables and returns them.
        * :py:attr:'output_manager' - Gets the outputs (via measurement or model)  and returns them.
        * :py:attr:'reward_manager' - Checks if the output bounds are exceeded and returns the results of the checks.
        Calculates the reward (which may depend on the results of the output bound checks) and returns it.
        * :py:attr:'scenario_manager' - Implements a scenario by changing the :py:attr:'config' of the other members.
        * :py:attr:'experiment_tracker' - Tracks a sequence of steps, e.g. by saving some variables in files or creating diagrams.

        The API methods of an environment are:
        * :py:meth:step - Updates the environment with actions returning observations and a reward.
        * :py:meth:reset - Resets the environment to initial process variable values returning initial observations and reward.
        Required before a sequence of steps.
        * :py:meth:close - Closes the environment. Important if the environment starts background processes that have to be shutdown.
        Thus, it is recommended to always call it if the environment is not needed anymore.

        .. note::
        The environment API works with Numpy (https://numpy.org/) arrays, whereas the member APIs work with dictionaries
        which have keys of type str and values of type float.
    """

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        """Configuration of environment."""
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
        """Processes the actions and disturbances into controls and dependent variables. Readonly.

        Checks if the control and dependent variable bounds would be exceeded
        if the actions received from the agent were applied blindly and returns the results of the checks.
        Based on the check results, calculates the controls and dependent variables and returns them.
        """
        pass

    @property
    @abstractmethod
    def output_manager(self) -> AbstractOutputManager:
        """Gets the outputs (via measurement or model) and returns them. Readonly."""
        pass

    @property
    @abstractmethod
    def reward_manager(self) -> AbstractObjectiveManager:
        """Calculates the reward. Readonly.

        Checks if the output bounds are exceeded and returns the results of the checks.
        Calculates the reward (which may depend on the results of the output bound checks) and returns it.
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
    def step(self, actions: dict) -> Tuple[np.array, float]:
        """Updates the environment with actions returning observations and a reward."""
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.array, float]:
        """Resets the environment to initial process variable values returning initial observations and reward.
        Required before a sequence of steps.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the environment.

        Important if the environment starts background processes that have to be shutdown.
        Thus, it is recommended to always call it if the environment is not needed anymore.
        """
        pass
