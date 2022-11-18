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

    # @abstractmethod
    # def setDisturbance(self, disturbanceSetting):
    #     pass

    @disturbanceSetting.setter
    @abstractmethod
    def disturbanceSetting(self, disturbanceSetting):
        pass

    # @abstractmethod
    # def setFibreMixture(self, fibreSetting):
    #     pass

    @fibreSetting.setter
    @abstractmethod
    def fibreSetting(self, fibreSetting):
        pass