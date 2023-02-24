from abc import ABC, abstractmethod


class AbstractScenarioManager(ABC):

    @property
    @abstractmethod
    def disturbanceSetting(self) -> dict:
        pass

    @property
    @abstractmethod
    def fibreSetting(self) -> dict:
        pass

    @disturbanceSetting.setter
    @abstractmethod
    def disturbanceSetting(self, disturbanceSetting):
        pass

    @fibreSetting.setter
    @abstractmethod
    def fibreSetting(self, fibreSetting):
        pass
