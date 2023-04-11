import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.reward import AbstractReward
from src.abstract_base_class.state_manager import AbstractStateManager
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker
from src.abstract_base_class.scenario_manager import AbstractScenarioManager


class TrainingEnvironment(AbstractTrainingEnvironment):
    def __init__(self, config: DictConfig, machine: AbstractModelWrapper, reward: AbstractReward,
                 stateManager: AbstractStateManager, experimentTracker: AbstractExperimentTracker,
                 scenarioManager: AbstractScenarioManager):

        self._machine = machine
        self._reward = reward
        self._experimentTracker = experimentTracker
        self._stateManager = stateManager
        self._scenarioManager = scenarioManager

        self._initialconfig = config.copy()
        self.reset()

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
    def stateManager(self) -> AbstractStateManager:
        return self._stateManager

    @property
    def experimentTracker(self) -> AbstractExperimentTracker:
        return self._experimentTracker

    @property
    def scenarioManager(self) -> AbstractScenarioManager:
        return self._scenarioManager

    @property
    def rewardRange(self) -> tuple[float, float]:
        return self._reward.reward_range

    @property
    def stepIndex(self):
        return self._stepIndex

    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        if self._status != "RUNNING":
            self._initExperiment()

        self._updateConfig()

        currentState, safetyFlag = self._stateManager.calculateControlsFromAction(action)

        observationArray, observationDict = self._machine.get_outputs(currentState)
        reward, reqsFlag = self._reward.calculateRewardAndReqsFlag(currentState, observationDict, safetyFlag)
        logVariables = \
            {"Reward": reward} | \
            {"Safety Flag": int(safetyFlag)} | \
            {"Requirements Flag": int(reqsFlag)} | \
            currentState | \
            observationDict
        self._experimentTracker.log(logVariables)
        info = dict()
        self._stepIndex = self._stepIndex + 1

        self._status = "RUNNING"

        return observationArray, reward, self._done, False, info

    def reset(self) -> tuple[np.array, dict]:
        self._experimentTracker.reset()
        self._reward.reset()
        initialState = self._stateManager.reset()
        self._machine.reset()
        self._scenarioManager.reset()

        self._stepIndex = 0
        self._done = False
        self._config = self._initialconfig.copy()

        observationArray, _ = self._machine.get_outputs(initialState)
        info = dict()

        self._status = "READY"
        return observationArray, info

    def render(self) -> None:
        pass

    def _initExperiment(self) -> None:
        self._experimentTracker.initRun()

    def _updateConfig(self) -> None:
        self._scenarioManager.update_requirements(self._stepIndex, self._reward.config.requirements)

        changed_outputs = self._scenarioManager.update_output_models(self._stepIndex, self._machine.config.outputModels)
        self._machine.update(changed_outputs)

        self._scenarioManager.update_disturbances(self._stepIndex, self._stateManager.config.disturbances)
