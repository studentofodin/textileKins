from abc import ABC, abstractmethod


class AbstractExperimentTracker(ABC):

    @property
    @abstractmethod
    def metrics(self) -> dict:
        pass

    @metrics.setter
    @abstractmethod
    def metrics(self, metrics):
        pass
        
    @abstractmethod
    def plotReward(self, rewardValue) -> bool:
        pass
        