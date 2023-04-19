from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractRewardManager(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @property
    @abstractmethod
    def reward_range(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def getReward(self, state: dict[str, float], outputs: dict[str, float],
                  safetyMet: bool) -> tuple[float, bool]:
        """
        determine reward value from state and outputs.
        also determine if outputs meet requirement constraints.
        return these values.
        if safety or requirement constraints are not met then fixed penalty is used for reward value.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        reset to initial values.
        """
        pass

    @abstractmethod
    def _reqsMet(self, outputs: dict[str, float]) -> bool:
        """
        check if outputs meet requirement constraints.
        """
        pass
