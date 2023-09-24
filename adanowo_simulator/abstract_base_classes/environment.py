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
    def disturbance_manager(self) -> AbstractDisturbanceManager:
        pass

    @property
    @abstractmethod
    def action_manager(self) -> AbstractActionManager:
        pass

    @property
    @abstractmethod
    def output_manager(self) -> AbstractOutputManager:
        pass

    @property
    @abstractmethod
    def reward_manager(self) -> AbstractObjectiveManager:
        pass

    @property
    @abstractmethod
    def scenario_manager(self) -> AbstractScenarioManager:
        pass

    @property
    @abstractmethod
    def experiment_tracker(self) -> AbstractExperimentTracker:
        pass

    @property
    @abstractmethod
    def step_index(self):
        pass

    @abstractmethod
    def step(self, actions: dict) -> Tuple[np.array, float]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.array, float]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
