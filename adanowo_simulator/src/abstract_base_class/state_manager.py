from abc import ABC, abstractmethod
from omegaconf import DictConfig
import numpy as np


class AbstractStateManager(ABC):
    @property
    @abstractmethod
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    @abstractmethod
    def config(self, c):
        self._config = c

    @property
    @abstractmethod
    def actionType(self) -> int:
        return self._actionType

    @property
    @abstractmethod
    def n_controls(self) -> int:
        return self._n_controls

    @property
    @abstractmethod
    def n_disturbances(self) -> int:
        return self._n_disturbances

    @abstractmethod
    def getState(self, action: np.array) -> tuple[dict[str, float], bool]:
        """
        calculate controls from action.
        the state is a concatenation of controls and disturbances (listed in own config).
        return state and if the action meets safety constraints.
        if the action does not meet safety constraints the controls remain the same as before.
        """
        pass

    @abstractmethod
    def reset(self) -> dict[str, float]:
        """
        reset to initial values.
        """

    @abstractmethod
    def _safetyMet(self, controls: dict[str, float]) -> bool:
        """
        check if controls meet safety constraints.
        """
        pass
