from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.reward_manager import AbstractRewardManager
from adanowo_simulator.abstract_base_classes.action_manager import AbstractActionManager
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
    def action_manager(self) -> AbstractActionManager:
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
    def step_index(self):
        pass

    @abstractmethod
    def step(self, actions: np.array) -> Tuple[np.array, float]:
        """
        run one timestep of the environment’s dynamics using the agent actions.
        return:
            observations (ObsType): observations due to the agent actions.
            reward (float): the reward as a result of taking the action.
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.array, float]:
        """
        resets the environment to an initial internal state, returning an initial observation
        and reward resulting from internal state.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        closes the environment.
        """
        pass

    @abstractmethod
    def _array_to_dict(self, array: np.array, keys: list[str]) -> dict[str, float]:
        """
        Converts a 1D array to a dict with given keys.
        """
        pass
