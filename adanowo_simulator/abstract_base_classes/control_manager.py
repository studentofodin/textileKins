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
    def step(self, actions: dict[str, float]) -> tuple[dict[str, float], dict[str, bool]]:
        """
        calculate controls from actions.
        return controls, if the control constraints are met with the given actions and actions as a dictionary.
        if the control constraints are not met the controls remain the same as before.
        """
        pass

    @abstractmethod
    def reset(self) -> tuple[dict[str, float], dict[str, bool]]:
        """
        reset to initial values.
        """


