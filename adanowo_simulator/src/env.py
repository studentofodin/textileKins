import numpy as np
from omegaconf import DictConfig
from gymnasium import spaces

from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.reward import AbstractReward
from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class TrainingEnvironment(AbstractTrainingEnvironment):
    def __init__(self, config: DictConfig, machine: AbstractModelWrapper, reward: AbstractReward,
                 safetyWrapper: AbstractSafetyWrapper, experimentTracker: AbstractExperimentTracker):

        self._config = config
        self._machine = machine
        self._reward = reward
        self._experimentTracker = experimentTracker
        self._safetyWrapper = safetyWrapper

        # set current controls and disturbances
        self._stepIndex = None
        self._done = False
        self._currentControls = dict()
        self._currentDisturbances = dict()
        self._currentState = dict()

        # set action space.
        self._actionSpace = spaces.Box(
            low=np.array([self._config.env_setup.actionSpace[param].low for
                          param in self._config.env_setup.usedControls], dtype=np.float32),
            high=np.array([self._config.env_setup.actionSpace[param].high for
                           param in self._config.env_setup.usedControls], dtype=np.float32),
            shape=(len(self._config.env_setup.usedControls),)
        )

        # set observation space.
        self._observationSpace = spaces.Box(
            low=np.array([self._config.env_setup.observationSpace[param].low for param in self._machine.output_names],
                         dtype=np.float32),
            high=np.array([self._config.env_setup.observationSpace[param].high for param in self._machine.output_names],
                          dtype=np.float32),
            shape=(len(self._machine.output_names),)
        )

        self._resetState()
        self._state = "NEW"

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
        return self._safetyWrapper

    @property
    def experimentTracker(self) -> AbstractExperimentTracker:
        return self._experimentTracker

    @property
    def currentControls(self) -> dict[str, float]:
        return self._currentControls

    @property
    def currentDisturbances(self) -> dict[str, float]:
        return self._currentDisturbances

    @property
    def actionSpace(self) -> np.array:
        return self._actionSpace

    @property
    def observationSpace(self) -> np.array:
        return self._observationSpace

    @property
    def rewardRange(self) -> tuple[float, float]:
        return self._reward.reward_range

    @property
    def stepIndex(self):
        return self._stepIndex

    def _resetState(self):
        self._stepIndex = 0
        for control in self._config.env_setup.usedControls:
            self._currentControls[control] = self._config.process_setup.initialControls[control]
        for disturbance in self._config.env_setup.usedDisturbances:
            self._currentDisturbances[disturbance] = self._config.process_setup.initialDisturbances[disturbance]
        self._currentState = self._currentControls | self._currentDisturbances

        if not self._safetyWrapper.safetyMet(self._currentControls):
            raise AssertionError("The initial setting is unsafe. Aborting Experiment.")

    def _initExperiment(self) -> None:
        self._experimentTracker.initTracker(self._config)

    def _calculateControlsFromAction(self, action: np.array) -> bool:
        updatedControls = dict()
        actionType = self._config.env_setup.actionType  # 0 for relative | 1 for absolute

        if actionType == 0:
            for index, key in enumerate(self._config.env_setup.usedControls):
                updatedControls[key] = self._currentControls[key] + action[index]

        else:
            for index, key in enumerate(self._config.env_setup.usedControls):
                updatedControls[key] = action[index]

        safetyFlag = self._safetyWrapper.safetyMet(updatedControls)
        if safetyFlag:
            self._currentControls = updatedControls
        self._currentState = self._currentControls | self._currentDisturbances

        return safetyFlag

    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        if self._state != "RUNNING":
            self._initExperiment()

        safetyFlag = self._calculateControlsFromAction(action)
        observationArray, observationDict = self._machine.get_outputs(self._currentState)
        reward, reqsFlag = self._reward.calculateRewardAndReqsFlag(self._currentState, observationDict, safetyFlag)
        logVariables = \
            {"Reward": reward} | \
            {"Safety Flag": int(safetyFlag)} | \
            {"Requirements Flag": int(reqsFlag)} | \
            self._currentControls | \
            self._currentDisturbances | \
            observationDict
        self._experimentTracker.log(logVariables)
        info = dict()
        self._stepIndex = self._stepIndex + 1

        self.render()
        self._state = "RUNNING"

        return observationArray, reward, self._done, False, info

    def reset(self) -> tuple[np.array, dict]:

        self._experimentTracker.finish_experiment()
        self._resetState()

        observationArray, _ = self._machine.get_outputs(self._currentState)
        info = dict()

        self._state = "READY"
        return observationArray, info

    def render(self) -> None:
        pass
