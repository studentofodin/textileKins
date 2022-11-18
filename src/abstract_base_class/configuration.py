from abc import ABC, abstractmethod

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
    
    @requirements.setter
    @abstractmethod
    def requirements(self, requirements):
        pass

    @actorConstraints.setter
    @abstractmethod
    def actorConstraints(self, actorConstraints):
        pass

    @productionScenario.setter
    @abstractmethod
    def productionScenario(self, productionScenario):
        pass

    @actionParams.setter
    @abstractmethod
    def actionParams(self, actionParams):
        pass

    @stateParams.setter
    @abstractmethod
    def stateParams(self, stateParams):
        pass

    @stepsUntilLabDataAvailable.setter
    @abstractmethod
    def stepsUntilLabDataAvailable(self, stepsUntilLabDataAvailable):
        pass

    @observationParams.setter
    @abstractmethod
    def observationParams(self, observationParams):
        pass
