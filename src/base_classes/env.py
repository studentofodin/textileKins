import numpy as np

from typing import Tuple
from gym import spaces
from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.base_classes.reward import Reward
from src.base_classes.experiment_tracker import ExperimentTracker
from src.base_classes.model_wrapper import ModelWrapper
from src.base_classes.safety_wrapper import SafetyWrapper


class TrainingEnvironment(AbstractTrainingEnvironment):
    """
    ### Action Space : Dimension is dynamic - to be imported from config

    ### Observation Space : Dimension is dynamic - to be imported from config

    ### Rewards : Calculated in Reward object after passing current state to Model

    ### Starting State : Machine starts at Initial State defined in Configuration
    """

    def __init__(
        self,
        config,
        machine: ModelWrapper,
        reward: Reward,
        experimentTracker: ExperimentTracker,
        safety: SafetyWrapper,
    ):
        self._config = config
        self._machine = machine
        self._reward = reward
        self._experimentTracker = experimentTracker
        self._safety = safety
        self._currentState = dict(config.initialState)
        self.actionParameters = config.actionParams
        self.observationParameters = config.observationParams
        self._actionSpace = spaces.Box(
            low=np.array(
                [
                    self.actionParameters[parameter].low
                    for parameter in self.actionParameters.list
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.actionParameters[parameter].high
                    for parameter in self.actionParameters.list
                ],
                dtype=np.float32,
            ),
            shape=(len(self.actionParameters.list),),
        )
        self._observationSpace = spaces.Box(
            low=np.array(
                [
                    self.observationParameters[parameter].low
                    for parameter in self.observationParameters.list
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.observationParameters[parameter].high
                    for parameter in self.observationParameters.list
                ],
                dtype=np.float32,
            ),
            shape=(len(self.observationParameters.list),),
        )
        self.done = False

    @property
    def machine(self):
        return self._machine

    @property
    def reward(self):
        return self._reward

    @property
    def experimentTracker(self):
        return self._experimentTracker

    @property
    def config(self):
        return self._config

    @property
    def safety(self):
        return self._safety

    @property
    def currentState(self) -> dict:
        return self._currentState

    @property
    def actionSpace(self) -> np.array:
        return self._actionSpace

    @property
    def observationSpace(self) -> np.array:
        return self._observationSpace

    @property
    def rewardRange(self) -> Tuple[float, float]:
        return (-float("inf"), float("inf"))

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
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
        safetyFlag = self.calculateStateFromAction(action)
        observationArray, observationDict = self.machine.get_outputs(self.currentState)
        reward = self.reward.calculateReward(self.currentState, observationDict, safetyFlag)
        self.done = False
        info = {}
        self.experimentTracker.log(reward, self.currentState, observationDict)
        return observationArray, reward, self.done, False, info

    def reset(self):
        # Reset Current State to Initial State
        # Reset required variables to start optimisation again
        # super().reset()
        self._currentState = dict(self.config.initialState)
        self._reward = 0.0
        self.done = False
        observationArray, _ = self.machine.get_outputs(self.currentState)
        info = {}
        return observationArray, info

    def calculateStateFromAction(self, action):
        updatedState = {}
        actionType = self.config.actionType # 0 for relative | 1 for absolute
        if actionType == 0: 
            for index, key in enumerate(self.actionParameters.list):
                updatedState[key] = self.currentState[key] + action[index]
        elif actionType == 1: 
            for index, key in enumerate(self.actionParameters.list):
                updatedState[key] = action[index]
        safetyFlag = self.safety.isWithinConstraints(updatedState)
        if safetyFlag:
            self._currentState = updatedState
        return safetyFlag

    def render(self):
        pass