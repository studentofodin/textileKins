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

    @abstractmethod
    def calculate_reward(self, state: dict[str, float], outputs: dict[str, float],
                         safety_met: bool) -> tuple[float, bool]:
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
    def _check_requirements(self, outputs: dict[str, float]) -> bool:
        """
        check if outputs meet requirement constraints.
        """
        pass

    @property
    @abstractmethod
    def reward_range(self) -> tuple[float, float]:
        """
        Return interval of possible reward values.
        """
        pass
