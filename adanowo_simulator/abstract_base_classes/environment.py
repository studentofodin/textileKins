from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.reward_manager import AbstractRewardManager
from adanowo_simulator.abstract_base_classes.control_manager import AbstractControlManager
from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager
from adanowo_simulator.abstract_base_classes.experiment_tracker import AbstractExperimentTracker
from adanowo_simulator.abstract_base_classes.scenario_manager import AbstractScenarioManager


class AbstractEnvironment(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @property
    @abstractmethod
    def reward_manager(self) -> AbstractRewardManager:
        pass

    @property
    @abstractmethod
    def control_manager(self) -> AbstractControlManager:
        pass

    @property
    @abstractmethod
    def disturbance_manager(self) -> AbstractDisturbanceManager:
        pass

    @property
    @abstractmethod
    def output_manager(self) -> AbstractOutputManager:
        pass

    @property
    @abstractmethod
    def experiment_tracker(self) -> AbstractExperimentTracker:
        pass

    @property
    @abstractmethod
    def scenario_manager(self) -> AbstractScenarioManager:
        pass

    @property
    @abstractmethod
    def reward_range(self) -> tuple[float, float]:
        pass

    @property
    @abstractmethod
    def step_index(self):
        pass

    @abstractmethod
    def step(self, actions: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """
        run one timestep of the environment’s dynamics using the agent actions.
        return:
            observations (ObsType): observations due to the agent actions.
            reward (float): the reward as a result of taking the action.
            terminated (bool): whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative.
                not needed here as there is no terminal state, thus always returned as False.
            truncated (bool) : whether the truncation condition outside the scope of the MDP is satisfied.
                typically, this is a timelimit, but could also indicate an agent physically going out of bounds.
                not needed here as there is no truncation condition, thus always returned as False.
            info (dict): contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.array, dict]:
        """
        resets the environment to an initial internal state, returning an initial observation and info.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        shuts down the environment.
        """
        pass

    @abstractmethod
    def _init_experiment(self) -> None:
        """
        initializes experiment.
        """
        pass

    @abstractmethod
    def _control_array_to_dict(self, array: np.array, keys: list[str]) -> dict[str, float]:
        """
        Converts a 1D array of control values to a dict of control values.
        This is necessary to keep the order of controls intact since dictionaries are unordered, unlike lists.
        """
        pass
