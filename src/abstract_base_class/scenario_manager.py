from abc import ABC, abstractmethod
import numpy as np

class AbstractScenarioManager(ABC):

    @property
    @abstractmethod
    def disturbanceSetting(self) -> dict:
        pass

    @property
    @abstractmethod
    def fibreSetting(self) -> dict:
        pass

    @abstractmethod
    def setDisturbance(self, disturbanceSetting):
        pass

    @abstractmethod
    def setFibreMixture(self, fibreSetting):
        pass