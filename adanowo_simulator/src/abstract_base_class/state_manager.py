from abc import ABC, abstractmethod
from omegaconf import DictConfig
import numpy as np


class AbstractStateManager(ABC):
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

    @property
    @abstractmethod
    def n_disturbances(self) -> int:
        pass

    @abstractmethod
    def get_state(self, action: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:
        """
        calculate controls from action.
        the state is a concatenation of controls and disturbances (listed in own config).
        return state, if the action meets safety constraints and action as a dictionary.
        if the action does not meet safety constraints the controls remain the same as before.
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
