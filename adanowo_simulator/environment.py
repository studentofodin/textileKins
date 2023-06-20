import numpy as np
from omegaconf import DictConfig
import logging
import sys

from adanowo_simulator.abstract_base_classes.environment import AbstractEnvironment
from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.reward_manager import AbstractRewardManager
from adanowo_simulator.abstract_base_classes.control_manager import AbstractControlManager
from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager
from adanowo_simulator.abstract_base_classes.experiment_tracker import AbstractExperimentTracker
from adanowo_simulator.abstract_base_classes.scenario_manager import AbstractScenarioManager

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(AbstractEnvironment):
    def __init__(self, config: DictConfig, output_manager: AbstractOutputManager, reward_manager: AbstractRewardManager,
                 control_manager: AbstractControlManager, disturbance_manager: AbstractDisturbanceManager,
                 experiment_tracker: AbstractExperimentTracker, scenario_manager: AbstractScenarioManager):
        self._output_manager = output_manager
        self._reward_manager = reward_manager
        self._experiment_tracker = experiment_tracker
        self._control_manager = control_manager
        self._disturbance_manager = disturbance_manager
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
    def output_manager(self) -> AbstractOutputManager:
        return self._output_manager

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
            logger.info("Experiment tracking has been initialized.")

        self._scenario_manager.step(self._step_index, self._disturbance_manager, self._output_manager,
                                    self._reward_manager)

        disturbances = self._disturbance_manager.step()
        controls, control_constraints_met, actions = self._control_manager.step(actions)

        outputs = self._output_manager.step(controls, disturbances)
        reward, output_constraints_met = self._reward_manager.step(controls, disturbances, outputs,
                                                                   control_constraints_met)

        log_variables = \
            {"Performance-Metrics": {"Reward": reward, "Control-Constraints-Met": int(control_constraints_met),
                                     "Output-Constraints-Met": int(output_constraints_met)},
             "Actions": actions,
             "Controls": controls,
             "Disturbances": disturbances,
             "Outputs": outputs}
        self._experiment_tracker.step(log_variables, self._step_index)

        info = dict()
        self._step_index = self._step_index + 1
        self._status = "RUNNING"

        observations = np.array(tuple(outputs.values()), dtype=np.float32)

        return observations, reward, False, False, info

    def reset(self) -> tuple[np.array, dict]:
        self._experiment_tracker.reset()
        self._reward_manager.reset()
        initial_controls = self._control_manager.reset()
        initial_disturbances = self._disturbance_manager.reset()
        self._output_manager.reset()
        self._scenario_manager.reset()

        self._step_index = 0
        self._config = self._initial_config.copy()

        outputs = self._output_manager.step(initial_controls, initial_disturbances)

        observations = np.array(tuple(outputs.values()), dtype=np.float32)

        info = dict()
        self._status = "READY"
        logger.info("Environment has been reset.")

        return observations, info

    def _init_experiment(self) -> None:
        self._experiment_tracker.init_experiment()
