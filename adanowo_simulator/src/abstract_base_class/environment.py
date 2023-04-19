from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.abstract_base_class.reward_manager import AbstractRewardManager
from src.abstract_base_class.state_manager import AbstractStateManager
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker
from src.abstract_base_class.scenario_manager import AbstractScenarioManager


class AbstractTrainingEnvironment(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @property
    @abstractmethod
    def machine(self) -> AbstractModelWrapper:
        pass

    @property
    @abstractmethod
    def rewardManager(self) -> AbstractRewardManager:
        pass

    @property
    @abstractmethod
    def stateManager(self) -> AbstractStateManager:
        pass

    @property
    @abstractmethod
    def experimentTracker(self) -> AbstractExperimentTracker:
        pass

    @property
    @abstractmethod
    def scenarioManager(self) -> AbstractScenarioManager:
        pass

    @property
    @abstractmethod
    def rewardRange(self) -> tuple[float, float]:
        pass

    @property
    @abstractmethod
    def stepIndex(self):
        pass

    @abstractmethod
    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """
        run one timestep of the environment’s dynamics using the agent actions.
        return:
            observation (ObsType): observation due to the agent actions.
            reward (float): the reward as a result of taking the action.
            terminated (bool): whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative.
                not needed here as there is no terminal state, thus always returned as False.
            truncated (bool) : whether the truncation condition outside the scope of the MDP is satisfied.
                typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                not needed here as there is no truncation condition, thus always returned as False.
            info (dict): contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.array, dict]:
        """
        resets the environment to an initial internal state, returning an initial observation and info.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        not needed here.
        """
        pass

    @abstractmethod
    def _initExperiment(self) -> None:
        """
        initializes experiment.
        """
        pass

    @abstractmethod
    def _updateConfigs(self) -> None:
        """
        updates the configs of itself, machine, reward, stateManager and experimentTracker primarily using the
        scenarioManager.
        """


