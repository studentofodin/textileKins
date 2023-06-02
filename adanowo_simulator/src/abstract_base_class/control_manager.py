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

    @property
    @abstractmethod
    def n_controls(self) -> int:
        pass

    @abstractmethod
    def get_controls(self, actions: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:
        """
        calculate controls from actions.
        return controls, if the actions meet safety constraints and actions as a dictionary.
        if the actions do not meet safety constraints the controls remain the same as before.
        """
        pass

    @abstractmethod
    def reset(self) -> dict[str, float]:
        """
        reset to initial values.
        """

    @abstractmethod
    def _control_constraints_met(self, controls: dict[str, float]) -> bool:
        """
        check if controls meet safety constraints.
        """
        pass
