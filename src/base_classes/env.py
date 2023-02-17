import numpy as np
from omegaconf import DictConfig
from typing import Tuple
from typing import Dict
from gym import spaces
import wandb as wb

from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.reward import AbstractReward
from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class TrainingEnvironment(AbstractTrainingEnvironment):
    def __init__(self, config: DictConfig, machine: AbstractModelWrapper, reward: AbstractReward,
                 safetyWrapper: AbstractSafetyWrapper, experimentTracker: AbstractExperimentTracker):

        self._stepIndex = 0

        self._config = config
        self._machine = machine
        self._reward = reward
        self._experimentTracker = experimentTracker
        self._safetyWrapper = safetyWrapper

        # set current controls.
        self._currentControls = dict()
        for control in self._config.usedControls:
            self._currentControls[control] = self._config.initialControls[control]

        # set current disturbances.
        self._currentDisturbances = dict()
        for disturbance in self._config.usedDisturbances:
            self._currentDisturbances[disturbance] = self._config.initialDisturbances[disturbance]

        currentState = self._currentControls | self._currentDisturbances
        observationArray, observationDict = self._machine.get_outputs(currentState)

        # set action space.
        self._actionSpace = spaces.Box(
            low=np.array([self._config.actionSpace[param].low for param in self._config.usedControls],
                         dtype=np.float32),
            high=np.array([self._config.actionSpace[param].high for param in self._config.usedControls],
                          dtype=np.float32),
            shape=(len(self._config.usedControls),)
        )

        # set observation space.
        self._observationSpace = spaces.Box(
            low=np.array([self._config.observationSpace[param].low for param in self._machine.output_names],
                         dtype=np.float32),
            high=np.array([self._config.observationSpace[param].high for param in self._machine.output_names],
                          dtype=np.float32),
            shape=(len(self._machine.output_names),)
        )

        self._done = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def machine(self) -> AbstractModelWrapper:
        return self._machine

    @property
    def reward(self) -> AbstractReward:
        return self._reward

    @property
    def safetyWrapper(self) -> AbstractSafetyWrapper:
        return self._safety

    @property
    def experimentTracker(self) -> AbstractExperimentTracker:
        return self._experimentTracker

    @property
    def currentControls(self) -> Dict[str, float]:
        return self._currentControls

    @property
    def currentDisturbances(self) -> Dict[str, float]:
        return self._currentDisturbances

    @property
    def actionSpace(self) -> np.array:
        return self._actionSpace

    @property
    def observationSpace(self) -> np.array:
        return self._observationSpace

    @property
    def rewardRange(self) -> Tuple[float, float]:
        return (-float("inf"), float("inf"))

    @property
    def stepIndex(self):
        pass

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:

        if self._stepIndex == 0:
            self.initExperiment()

        safetyFlag = self.calculateControlsFromAction(action)
        currentState = self._currentControls | self._currentDisturbances
        observationArray, observationDict = self._machine.get_outputs(currentState)
        reward, reqsFlag = self._reward.calculateRewardAndReqsFlag(currentState, observationDict, safetyFlag)
        logVariables = {"Reward": reward} | {"Safety Flag": int(safetyFlag)} | {"Requirements Flag": int(reqsFlag)} | \
                       self._currentControls | self._currentDisturbances | observationDict
        self._experimentTracker.log(logVariables)

        self._done = False
        info = dict()
        self._stepIndex = self._stepIndex + 1

        return observationArray, reward, self._done, False, info

    def initExperiment(self) -> None:
        self._experimentTracker.initWandB()
        safetyFlag = self._safetyWrapper.safetyMet(self._currentControls)
        currentState = self._currentControls | self._currentDisturbances
        reward, reqsFlag = self._reward.calculateRewardAndReqsFlag(currentState, self._machine.outputs, safetyFlag)
        logVariables = {"Reward": reward} | {"Safety Flag": int(safetyFlag)} | {"Requirements Flag": int(reqsFlag)} | \
                       self._currentControls | self._currentDisturbances | self._machine.outputs
        self._experimentTracker.log(logVariables)

    def reset(self) -> Tuple[np.array, dict]:

        wb.finish()
        self._stepIndex = 0

        # set current controls.
        self._currentControls = dict()
        for control in self._config.usedControls:
            self._currentControls[control] = self._config.initialControls[control]

        # set current disturbances.
        self._currentDisturbances = dict()
        for disturbance in self._config.usedDisturbances:
            self._currentDisturbances[disturbance] = self._config.initialDisturbances[disturbance]

        currentState = self._currentControls | self._currentDisturbances
        observationArray, _ = self._machine.get_outputs(currentState)

        info = dict()

        return observationArray, info


    def calculateControlsFromAction(self, action: np.array) -> bool:

        updatedControls = dict()
        actionType = self._config.actionType # 0 for relative | 1 for absolute

        if actionType == 0: 
            for index, key in enumerate(self._config.usedControls):
                updatedControls[key] = self._currentControls[key] + action[index]

        else:
            for index, key in enumerate(self._config.usedControls):
                updatedControls[key] = action[index]

        safetyFlag = self._safetyWrapper.safetyMet(updatedControls)
        if safetyFlag:
            self._currentControls = updatedControls

        return safetyFlag

    def render(self) -> None:
        pass