from abc import ABC, abstractmethod
import numpy as np

class AbstractExperimentTracking(ABC):

    @property
    @abstractmethod
    def metric(self) -> dict:
        pass

    @abstractmethod
    def plotReward(self, rewardValue) -> bool:
        pass

class ExperimentTracking(AbstractExperimentTracking):

    def __init__(self, metric):
        self._metric = metric     

    def plotReward(self, rewardValue:float):
        return True

    @property
    def metric(self):
        return self._metric

def main():
    e= ExperimentTracking({'speed':4, 'speed2':5})
    print(e.metric)
    print(e.plotReward(5.0))

main()
        