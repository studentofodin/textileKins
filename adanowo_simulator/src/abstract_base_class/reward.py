from abc import ABC, abstractmethod


class AbstractReward(ABC):

    @property
    @abstractmethod
    def rewardValue(self) -> float:
        pass

    @property
    @abstractmethod
    def reqsFlag(self) -> bool:
        pass

    @property
    @abstractmethod
    def reward_range(self) -> (float, float):
        pass

    @abstractmethod
    def reqsMet(self, outputs: dict[str, float]) -> bool:
        """
        check if the outputs are within requirement constraints.
        return this value and assign it to property reqsFlag.
        used within calculateReward().
        """
        pass

    @abstractmethod
    def calculateRewardAndReqsFlag(self, state: dict[str, float], outputs: dict[str, float],
                                   safetyFlag: bool) -> tuple[float, bool]:
        """
        determine reward value from state and outputs.
        also determine requirements flag.
        return these values and assign them to properties rewardValue and reqsFlag.
        if safety flag or requirements flag is true then fixed penalty is used for reward value.
        """
        pass
