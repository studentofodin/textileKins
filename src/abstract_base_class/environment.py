from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from src.abstract_base_class.reward import AbstractReward
from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class AbstractTrainingEnvironment(ABC):

    @property
    @abstractmethod
    def config(self):
        pass

    @property
    @abstractmethod
    def state(self) -> dict:
        pass

    @property
    @abstractmethod
    def actionsAreSafe(self) -> bool:
        pass

    @property
    @abstractmethod
    def reward(self) -> AbstractReward:
        pass


    @property
    @abstractmethod
    def machine(self) -> AbstractModelWrapper:
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
    def mapActionsToStates(actions: dict) -> None:
        pass

    @abstractmethod
    def step(self, actions: dict) -> Tuple[np.array, float, bool, bool, dict]:
        """
        Returns:

        observation(object) – this will be an element of the environment’s observation_space.This may, for instance,
        be a numpy array containing the positions and velocities of certain objects.

        reward(float) – The amount of reward returned as a result of taking the action.

        terminated(bool) – whether a terminal state (as defined under the MDP of the task) is reached. In this case
        further step() calls could return undefined results.

        truncated(bool) – whether a truncation condition outside the scope of the MDP is satisfied. Typically a
        timelimit, but could also be used to indicate agent physically going out of bounds. Can be used to end the
        episode prematurely before a terminal state is reached.

        done(bool) – A boolean value for if the episode has ended, in which case further step() calls will return
        undefined results. A done signal may be emitted for different reasons: Maybe the task underlying the
        environment was solved successfully, a certain timelimit was exceeded, or the physics simulation has entered an
        invalid state.

        info(dictionary) – info contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        This might, for instance, contain: metrics that describe the agent’s performance state, variables that are
        hidden from observations, or individual reward terms that are combined to produce the total reward. It also
        can contain information that distinguishes truncation and termination, however this is deprecated in favour of
        returning two booleans, and will be removed in a future version.
        """

    @abstractmethod
    def reset(self) -> Tuple[np.array, dict]:
        """
        Returns

        observation (object) – Observation of the initial state. This will be an element of observation_space
        (typically a numpy array) and is analogous to the observation returned by step().

        info (dictionary) – This dictionary contains auxiliary information complementing observation.
        It should be analogous to the info returned by step().
        """
        pass
