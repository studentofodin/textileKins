from abc import ABC, abstractmethod
import numpy as np
import gym

from gym import error, logger, spaces
from gym.spaces import Space

from experiment_tracking import AbstractExperimentTracking
from reward import AbstractReward
from interfaces import ModelInterface



class AbstractITAEnvironment(ABC, gym.Env):

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
    def current_state(self) -> np.array:
        pass

    # @abstractmethod
    # def define_state_space(self):
    #     pass

    # @abstractmethod
    # def step(self, action) -> (observation:int, reward:int, termination:int, truncation:int, info:int) :
    #     pass

    # @abstractmethod
    # def define_action_space(self):
    #     pass

    # @abstractmethod
    # def reset(self):
    #     pass


    # ai gym wrapper , or inherit, implement class keeping type same but with dummy values (machine model - arrays of 1 for ex.)
    # implement functions of env not other classes
    # check ai gym env implementation