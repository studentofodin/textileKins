from abc import ABC, abstractmethod
from typing import Tuple
from typing import Dict
from omegaconf import DictConfig


class AbstractReward(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @property
    @abstractmethod
    def rewardValue(self) -> float:
        pass

    @property
    @abstractmethod
    def reqsFlag(self) -> bool:
        pass

    @abstractmethod
    def reqsMet(self, outputs: Dict[str, float]) -> bool:
        """
        check if the outputs are within requirement constraints.
        return this value and assign it to property reqsFlag.
        used within calculateReward().
        """
        pass

    @abstractmethod
    def calculateRewardAndReqsFlag(self, state: Dict[str, float], outputs: Dict[str, float],
                                   safetyFlag: bool) -> Tuple[float, bool]:
        """
        determine reward value from state and outputs.
        also determine requirements flag.
        return these values and assign them to properties rewardValue and reqsFlag.
        if safety flag or requirements flag is true then fixed penalty is used for reward value.
        """
        pass
