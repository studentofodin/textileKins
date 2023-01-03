from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker
from src.abstract_base_class.reward import AbstractReward
from src.abstract_base_class.model_wrapper import AbstractModelWrapper


class AbstractTrainingEnvironment(ABC):

    @property
    @abstractmethod
    def reward(self) -> AbstractReward:
        pass

    @property
    @abstractmethod
    def experimentTracker(self) -> AbstractExperimentTracker:
        pass

    @property
    @abstractmethod
    def machine(self) -> AbstractModelWrapper:
        pass

    @property
    @abstractmethod
    def currentState(self) -> np.array:
        pass

    @property
    @abstractmethod
    def actionSpace(self) -> np.array:
        pass

    @property
    @abstractmethod
    def observationSpace(self) -> np.array:
        pass

    @property
    @abstractmethod
    def rewardRange(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        pass

    @abstractmethod
    def reset(self):
        pass
