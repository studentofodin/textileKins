import numpy as np
from omegaconf import DictConfig
from typing import List

from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.reward import AbstractReward
from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker
from src.abstract_base_class.scenario_manager import AbstractScenarioManager


class TrainingEnvironment(AbstractTrainingEnvironment):
    def __init__(self, config: DictConfig, machine: AbstractModelWrapper, reward: AbstractReward,
                 safetyWrapper: AbstractSafetyWrapper, experimentTracker: AbstractExperimentTracker,
                 scenarioManager: AbstractScenarioManager, actionType: int):

        self._initialconfig = config.copy()
        self._config = config
        self._machine = machine
        self._reward = reward
        self._experimentTracker = experimentTracker
        self._safetyWrapper = safetyWrapper
        self._scenarioManager = scenarioManager
        self._actionType = actionType

        # set current controls and disturbances
        self._stepIndex = None
        self._done = False
        self._currentControls = dict()
        self._currentState = dict()
        self._resetState()
        self._state = "NEW"

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

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
    def scenarioManager(self) -> AbstractScenarioManager:
        return self._scenarioManager

    @property
    def currentControls(self) -> dict[str, float]:
        return self._currentControls

    @property
    def rewardRange(self) -> tuple[float, float]:
        return self._reward.reward_range

    @property
    def stepIndex(self):
        return self._stepIndex

    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        if self._state != "RUNNING":
            self._initExperiment()

        self._update()

        safetyFlag = self._calculateControlsFromAction(action)

        observationArray, observationDict = self._machine.get_outputs(self._currentState)
        reward, reqsFlag = self._reward.calculateRewardAndReqsFlag(self._currentState, observationDict, safetyFlag)
        logVariables = \
            {"Reward": reward} | \
            {"Safety Flag": int(safetyFlag)} | \
            {"Requirements Flag": int(reqsFlag)} | \
            self._currentState | \
            observationDict
        self._experimentTracker.log(logVariables)
        info = dict()
        self._stepIndex = self._stepIndex + 1

        self._state = "RUNNING"

        return observationArray, reward, self._done, False, info

    def reset(self) -> tuple[np.array, dict]:
        self._experimentTracker.finish_experiment()

        self._config = self._initialconfig.copy()
        self._experimentTracker.wb_config = self._config.experimentTracker
        self._reward.config = self._config.product_setup
        self._safetyWrapper.config = self._config.process_setup
        self._machine.config = self._config.env_setup
        self._scenarioManager.config = self._config.scenario_setup

        self._machine.update(self._config.env_setup.usedOutputs)
        self._resetState()

        observationArray, _ = self._machine.get_outputs(self._currentState)
        info = dict()

        self._state = "READY"
        return observationArray, info

    def render(self) -> None:
        pass

    def _initExperiment(self) -> None:
        self._experimentTracker.initTracker(self._config)

    def _calculateControlsFromAction(self, action: np.array) -> bool:
        updatedControls = dict()

        if self._actionType == 0:
            for index, key in enumerate(self._config.env_setup.usedControls):
                updatedControls[key] = self._currentControls[key] + action[index]

        else:
            for index, key in enumerate(self._config.env_setup.usedControls):
                updatedControls[key] = action[index]

        safetyFlag = self._safetyWrapper.safetyMet(updatedControls)
        if safetyFlag:
            self._currentControls = updatedControls
        self._currentState = self._currentControls | dict(self._config.process_setup.disturbances)

        return safetyFlag

    def _update(self) -> None:
        self._scenarioManager.update_requirements(self._stepIndex, self._reward._config.requirements)

        changed_outputs = self._scenarioManager.update_output_models(self._stepIndex, self._machine._config.outputModels)
        self._machine.update(changed_outputs)

        changed_disturbances = self._scenarioManager.update_disturbances(self._stepIndex, self._config.process_setup.disturbances)

    def _resetState(self):
        self._stepIndex = 0

        self._currentControls = dict(self._config.process_setup.initialControls)
        self._currentState = self._currentControls | dict(self._config.process_setup.disturbances)

        if not self._safetyWrapper.safetyMet(self._currentControls):
            raise AssertionError("The initial setting is unsafe. Aborting Experiment.")
