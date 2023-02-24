from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Dict


class AbstractSafetyWrapper(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @property
    @abstractmethod
    def safetyFlag(self) -> bool:
        pass

    @abstractmethod
    def safetyMet(self, controls: Dict[str, float]) -> bool:
        """
        check if the controls are within safety constraints.
        return this value and assign it to property safetyFlag.
        """
        pass

