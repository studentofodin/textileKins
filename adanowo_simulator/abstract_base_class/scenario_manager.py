from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.abstract_base_class.output_manager import AbstractOutputManager
from src.abstract_base_class.reward_manager import AbstractRewardManager
from src.abstract_base_class.control_manager import AbstractControlManager
from src.abstract_base_class.disturbance_manager import AbstractDisturbanceManager


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
    def step(self, step_index: int, *managers: AbstractControlManager | AbstractDisturbanceManager | \
             AbstractOutputManager | AbstractRewardManager) -> None:
        """
        update the configs of the managers according to own config.
        """
        pass

    def reset(self) -> None:
        """
        reset to initial values.
        """
        pass
