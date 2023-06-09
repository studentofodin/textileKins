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
    def step(self, controls: dict[str, float], disturbances: dict[str, float], outputs: dict[str, float],
                   control_constraints_met: bool) -> tuple[float, bool]:
        """
        determine reward value from state and outputs.
        also determine if output constraints are met.
        return these values.
        if control or output constraints are not met then fixed penalty is used for reward value.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        reset to initial values.
        """
        pass

    @abstractmethod
    def _get_reward(self, controls: dict[str, float], disturbances: dict[str, float], outputs: dict[str, float]) -> float:
        """
        determines reward.
        """
        pass

    @abstractmethod
    def _output_constraints_met(self, outputs: dict[str, float]) -> bool:
        """
        check if outputs constraints are met.
        """
        pass

    @property
    @abstractmethod
    def reward_range(self) -> tuple[float, float]:
        """
        Return interval of possible reward values.
        """
        pass
