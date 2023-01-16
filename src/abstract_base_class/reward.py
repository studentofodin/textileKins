from abc import ABC, abstractmethod


class AbstractReward(ABC):

    @property
    @abstractmethod
    def config(self) -> dict:
        pass

    @property
    @abstractmethod
    def requirements(self) -> dict:
        pass

    @property
    @abstractmethod
    def weights(self) -> dict:
        pass

    @property
    @abstractmethod
    def rewardValue(self) -> float:
        pass

    @abstractmethod
    def calculateReward(self, state: dict, observation: dict, safety_flag: bool) -> float:
        pass
