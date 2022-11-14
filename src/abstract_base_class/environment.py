from abc import ABC, abstractmethod
import numpy as np
from abstract_base_class.experiment_tracking import AbstractExperimentTracking
from abstract_base_class.reward import AbstractReward
from abstract_base_class.interfaces import ModelInterface



class AbstractITAEnvironment(ABC):

    @property
    @abstractmethod
    def machine(self) -> bool:
        pass

    @property
    @abstractmethod
    def reward(self) -> AbstractReward:
        pass

    @property
    @abstractmethod
    def tracker(self) -> AbstractExperimentTracking:
        pass

    @property
    @abstractmethod
    def machine(self) -> ModelInterface:
        pass

    @property
    @abstractmethod
    def max_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def current_state(self) -> current_state:
        pass

    @abstractmethod
    def define_state_space(self):
        pass

    @abstractmethod
    def step(self, action:) -> observation, reward, termination, truncation, info :
        pass

    @abstractmethod
    def define_action_space(self):
        pass

    @abstractmethod
    def reset(self):
        pass