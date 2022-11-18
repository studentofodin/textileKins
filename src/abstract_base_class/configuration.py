from abc import ABC, abstractmethod
import numpy as np

class AbstractConfiguration(ABC):

    @property
    @abstractmethod
    def requirements(self) -> dict:
        pass

    @property
    @abstractmethod
    def actorConstraints(self) -> dict:
        pass

    @property
    @abstractmethod
    def productionScenario(self) -> dict:
        pass

    @property
    @abstractmethod
    def actionParams(self) -> dict:
        pass

    @property
    @abstractmethod
    def stateParams(self) -> dict:
        pass

    @property
    @abstractmethod
    def stepsUntilLabDataAvailable(self) -> int:
        pass

    @property
    @abstractmethod
    def observationParams(self) -> dict:
        pass  