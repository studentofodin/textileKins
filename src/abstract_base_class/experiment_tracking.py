from abc import ABC, abstractmethod
import numpy as np

class AbstractExperimentTracking(ABC):

    @property
    @abstractmethod
    def metric(self) -> dict:
        pass

    @abstractmethod
    def plot_reward(self, reward_value) -> bool:
        pass
