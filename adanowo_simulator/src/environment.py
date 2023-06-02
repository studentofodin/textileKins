import numpy as np
from omegaconf import DictConfig
import logging
import sys

from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.abstract_base_class.model_manager import AbstractModelManager
from src.abstract_base_class.reward_manager import AbstractRewardManager
from src.abstract_base_class.state_manager import AbstractStateManager
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker
from src.abstract_base_class.scenario_manager import AbstractScenarioManager


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingEnvironment(AbstractTrainingEnvironment):
    def __init__(self, config: DictConfig, machine: AbstractModelManager, reward_manager: AbstractRewardManager,
                 state_manager: AbstractStateManager, experiment_tracker: AbstractExperimentTracker,
                 scenario_manager: AbstractScenarioManager):

        self._machine = machine
        self._reward_manager = reward_manager
        self._experiment_tracker = experiment_tracker
        self._state_manager = state_manager
        self._scenario_manager = scenario_manager

        self._initial_config = config.copy()
        self._config = config.copy()
        self._step_index = 0
        self._status = "READY"

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def machine(self) -> AbstractModelManager:
        return self._machine

    @property
    def reward_manager(self) -> AbstractRewardManager:
        return self._reward_manager

    @property
    def state_manager(self) -> AbstractStateManager:
        return self._state_manager

    @property
    def experiment_tracker(self) -> AbstractExperimentTracker:
        return self._experiment_tracker

    @property
    def scenario_manager(self) -> AbstractScenarioManager:
        return self._scenario_manager

    @property
    def reward_range(self) -> tuple[float, float]:
        return self._reward_manager.reward_range

    @property
    def step_index(self):
        return self._step_index

    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        if self._status != "RUNNING":
            self._init_experiment()
            logger.info("Experiment tracking has been initialized.")

        self._update_configs()

        state, safety_met, action_dict = self._state_manager.update_state(action)

        outputs_array, outputs_dict = self._machine.get_model_outputs(state)
        reward, reqs_met = self._reward_manager.calculate_reward(state, outputs_dict, safety_met)

        log_variables = \
            {"Reward": reward} | \
            {"Safety Met": int(safety_met)} | \
            {"Requirements Met": int(reqs_met)} | \
            action_dict | \
            state | \
            outputs_dict
        self._experiment_tracker.log(log_variables)

        info = dict()
        self._step_index = self._step_index + 1
        self._status = "RUNNING"

        return outputs_array, reward, False, False, info

    def reset(self) -> tuple[np.array, dict]:
        self._experiment_tracker.reset()
        self._reward_manager.reset()
        initial_state = self._state_manager.reset()
        self._machine.reset()
        self._scenario_manager.reset()

        self._step_index = 0
        self._config = self._initial_config.copy()

        observation_array, _ = self._machine.get_model_outputs(initial_state)
        info = dict()

        self._status = "READY"
        logger.info("Environment has been reset.")
        return observation_array, info

    def _init_experiment(self) -> None:
        self._experiment_tracker.init_run()

    def _update_configs(self) -> None:
        self._reward_manager.config.requirements = \
            self._scenario_manager.update_requirements(self._step_index,
                                                       self._reward_manager.config.requirements.copy())

        self._machine.config.output_models, changed_outputs = \
            self._scenario_manager.update_model_allocation(self._step_index, self._machine.config.output_models.copy())
        self._machine.update_model_allocation(changed_outputs)

        self._state_manager.config.disturbances = \
            self._scenario_manager.update_disturbances(self._step_index, self._state_manager.config.disturbances.copy())
