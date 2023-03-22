from abc import ABC, abstractmethod
from typing import Tuple
from typing import Dict
import numpy as np

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.reward import AbstractReward
from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class AbstractTrainingEnvironment(ABC):

    @property
    @abstractmethod
    def config(self):
        pass

    @property
    @abstractmethod
    def machine(self) -> AbstractModelWrapper:
        pass

    @property
    @abstractmethod
    def reward(self) -> AbstractReward:
        pass

    @property
    @abstractmethod
    def safetyWrapper(self) -> AbstractSafetyWrapper:
        pass

    @property
    @abstractmethod
    def experimentTracker(self) -> AbstractExperimentTracker:
        pass

    @property
    @abstractmethod
    def currentControls(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def currentDisturbances(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def rewardRange(self) -> Tuple[float, float]:
        pass

    @property
    @abstractmethod
    def stepIndex(self):
        pass

    @abstractmethod
    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        """
        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool)/truncated (bool) : not needed as safety limits checked by safety wrapper
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
            done (bool): Not needed since it is a continuous process
        """
        pass

    @abstractmethod
    def _initExperiment(self) -> None:
        """
        is executed in first step of an experiment.
        sets several values to initial values.
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.array, dict]:
        """
        reset controls and disturbances to initial.
        determine observation based on these and return it.
        """
        pass

    @abstractmethod
    def _calculateControlsFromAction(self, action: np.array) -> bool:
        """
        calculates controls from action and assigns value to property currentControls.
        also determines if action leads to safe controls and returns this value as safetyFlag.
        """
        pass
