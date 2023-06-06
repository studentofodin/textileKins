import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.environment import AbstractTrainingEnvironment
from src.abstract_base_class.output_manager import AbstractOutputManager
from src.abstract_base_class.reward_manager import AbstractRewardManager
from src.abstract_base_class.control_manager import AbstractControlManager
from src.abstract_base_class.disturbance_manager import AbstractDisturbanceManager
from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker
from src.abstract_base_class.scenario_manager import AbstractScenarioManager


class TrainingEnvironment(AbstractTrainingEnvironment):
    def __init__(self, config: DictConfig, machine: AbstractOutputManager, reward_manager: AbstractRewardManager,
                 control_manager: AbstractControlManager, disturbance_manager: AbstractDisturbanceManager,
                 experiment_tracker: AbstractExperimentTracker, scenario_manager: AbstractScenarioManager):

        self._machine = machine
        self._reward_manager = reward_manager
        self._experiment_tracker = experiment_tracker
        self._control_manager = control_manager
        self._disturbance_manager = disturbance_manager
        self._scenario_manager = scenario_manager

        self._initial_config = config.copy()
        self._config = None
        self._step_index = None
        self._status = None
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def machine(self) -> AbstractOutputManager:
        return self._machine

    @property
    def reward_manager(self) -> AbstractRewardManager:
        return self._reward_manager

    @property
    def control_manager(self) -> AbstractControlManager:
        return self._control_manager

    @property
    def disturbance_manager(self) -> AbstractDisturbanceManager:
        return self._disturbance_manager

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

    def step(self, actions: np.array) -> tuple[np.array, float, bool, bool, dict]:
        if self._status != "RUNNING":
            self._init_experiment()

        self._update_configs()

        disturbances = self._disturbance_manager.get_disturbances()
        controls, safety_met, actions_dict = self._control_manager.get_controls(actions)
        states = controls | disturbances


        outputs_array, outputs_dict = self._machine.get_outputs(states)
        reward, reqs_met = self._reward_manager.get_reward(states, outputs_dict, safety_met)

        log_variables = \
            {"Reward": reward} | \
            {"Safety Met": int(safety_met)} | \
            {"Requirements Met": int(reqs_met)} | \
            actions_dict | \
            states | \
            outputs_dict
        self._experiment_tracker.log(log_variables)

        info = dict()
        self._step_index = self._step_index + 1
        self._status = "RUNNING"

        return outputs_array, reward, False, False, info

    def reset(self) -> tuple[np.array, dict]:
        self._experiment_tracker.reset()
        self._reward_manager.reset()
        initial_controls = self._control_manager.reset()
        initial_disturbances = self._disturbance_manager.reset()
        initial_states = initial_controls | initial_disturbances
        self._machine.reset()
        self._scenario_manager.reset()

        self._step_index = 0
        self._config = self._initial_config.copy()

        observation_array, _ = self._machine.get_outputs(initial_states)
        info = dict()

        self._status = "READY"
        return observation_array, info

    def _init_experiment(self) -> None:
        self._experiment_tracker.init_run()

    def _update_configs(self) -> None:
        self._reward_manager.config.output_bounds = \
            self._scenario_manager.update_output_bounds(self._step_index,
                                                       self._reward_manager.config.output_bounds.copy())

        self._machine.config.output_models, changed_outputs = \
            self._scenario_manager.update_output_models(self._step_index, self._machine.config.output_models.copy())
        self._machine.update(changed_outputs)

        self._control_manager.config.disturbances = \
            self._scenario_manager.update_disturbances(self._step_index, self._disturbance_manager.config.disturbances.copy())
