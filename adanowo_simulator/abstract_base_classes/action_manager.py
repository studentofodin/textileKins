from abc import ABC, abstractmethod

from omegaconf import DictConfig


class AbstractActionManager(ABC):
    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def step(self, actions: dict[str, float], disturbances: dict[str, float] | None) -> tuple[dict[str, float], dict[str, bool]]:
        """
        calculate controls from actions.
        return controls, if the control constraints are met with the given actions and actions as a dictionary.
        if the control constraints are not met the controls remain the same as before.
        """
        pass

    @abstractmethod
    def reset(self, disturbances: dict[str, float] | None) -> tuple[dict[str, float], dict[str, bool]]:
        """
        reset to initial values.
        """


