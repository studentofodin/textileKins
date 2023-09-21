from abc import ABC, abstractmethod

from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager
from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.reward_manager import AbstractRewardManager


class AbstractScenarioManager(ABC):
    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, step_index: int, disturbance_manager: AbstractDisturbanceManager,
             output_manager: AbstractOutputManager, reward_manager: AbstractRewardManager) -> None:
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
