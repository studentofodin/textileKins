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
    def step_index(self):
        pass

    @abstractmethod
    def step(self, actions: np.array) -> Tuple[np.array, float]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.array, float]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def _array_to_dict(self, array: np.array, keys: list[str]) -> dict[str, float]:
        pass
