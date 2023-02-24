from typing import Tuple, List
from gym import spaces
import numpy as np

from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.abstract_base_class.reward import AbstractReward
from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker

class TrainingEnvironment(AbstractTrainingEnvironment):

    def __init__(self, config, reward: AbstractReward, machine: AbstractModelWrapper,
                 experimentTracker: AbstractExperimentTracker, initialState: dict, actionsAreAbsolute: bool):
        self._config = config
        self._reward = reward
        self._machine = machine
        self._experimentTracker = experimentTracker
        self._initialState = initialState
        self._state = initialState
        self._actionsAreAbsolute = actionsAreAbsolute
        self._actionsAreSafe = True
        self._actionSpace = spaces.Box(low=np.array([0,0,0]), high=np.array([100,100,100]), shape=(3,), dtype=np.float64)
        self._observationSpace = spaces.Box(low=np.array([0,0]), high=np.array([50,50]), shape=(2,), dtype=np.float64)

    @property
    def config(self):
        return self._config

    @property
    def state(self) -> dict:
        return self._state

    @property
    def actionsAreSafe(self) -> bool:
        return self._actionsAreSafe

    @property
    def reward(self) -> AbstractReward:
        return self._reward

    @property
    def machine(self) -> AbstractModelWrapper:
        return self._machine

    @property
    def safetyWrapper(self) -> AbstractSafetyWrapper:
        return self._safetyWrapper

    @property
    def experimentTracker(self) -> AbstractExperimentTracker:
        return self._experimentTracker

    @property
    def actionSpace(self) -> np.array:
        return self._actionSpace

    @property
    def observationSpace(self) -> np.array:
        return self._observationSpace

    @property
    def rewardRange(self) -> Tuple[float, float]:
        return (-float("inf"), float("inf"))

    def mapActionsToState(self, actions: dict) -> None:
        if list(actions.keys()) != list(self._state.keys()):
            raise Exception('actions and state do not have the same keys')

        if self._actionsAreAbsolute:
            if self._safetyWrapper.isWithinConstraints(actions):
                self._state = actions
                self._actionsAreSafe = True
            else:
                self._actionsAreSafe = False

        else:
            newState = {k: actions.get(k) + self._state.get(k) for k in actions.keys()}
            if self._safetyWrapper.isWithinConstraints(newState):
                self._state = newState
                self.actionsAreSafe = True
            else:
                self._actionsAreSafe = False


    def step(self, actions: dict) -> Tuple[np.array, float, bool, bool, dict]:
        observation = self._machine.getOutput(actions)
        reward = self._reward.calculateReward(self._state, observation, not self._actionsAreSafe)
        terminated = False
        truncated = False
        info = dict()
        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[np.array, dict]:
        self._state = self._initialState
        self._reward = 0.0
        observation = self._machine.getOutput(self._initialState)
        info = dict()
        return observation, info
