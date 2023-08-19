from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractRewardManager(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c) -> None:
        pass

    @abstractmethod
    def step(self, state: dict[str, float], outputs: dict[str, float], control_constraints_met: dict[str, bool]) -> \
            tuple[float, dict[str, bool]]:
        pass

    @abstractmethod
    def reset(self, state: dict[str, float], outputs: dict[str, float], control_constraints_met: dict[str, bool]) -> \
            tuple[float, dict[str, bool]]:
        pass
