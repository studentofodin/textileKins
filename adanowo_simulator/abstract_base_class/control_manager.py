from abc import ABC, abstractmethod
from omegaconf import DictConfig
import numpy as np


class AbstractControlManager(ABC):
    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, actions: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:
        """
        calculate controls from actions.
        return controls, if the control constraints are met with the given actions and actions as a dictionary.
        if the control constraints are not met the controls remain the same as before.
        """
        pass

    @abstractmethod
    def reset(self) -> dict[str, float]:
        """
        reset to initial values.
        """


