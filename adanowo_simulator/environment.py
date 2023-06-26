import numpy as np
from omegaconf import DictConfig, OmegaConf
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
        self._config = None
        self._step_index = None
        self._ready = False
        logger.info("Environment has been created.")

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

    def _array_to_dict(self, array: np.array, keys: list[str]) -> dict[str, float]:
        dictionary = dict()
        for index, key in enumerate(keys):
            dictionary[key] = array[index]
        return dictionary

    def step(self, actions_array: np.array) -> tuple[np.array, float, bool, bool, dict]:
        if self._ready:
            try:
                if self._step_index == 0:
                    logger.info("Experiment is running.")
                actions = self._array_to_dict(actions_array, OmegaConf.to_container(self._config.used_primary_controls))
                self._scenario_manager.step(self._step_index, self._disturbance_manager, self._output_manager,
                                            self._reward_manager)
                disturbances = self._disturbance_manager.step()
                controls, control_constraints_met = self._control_manager.step(actions, disturbances)
                outputs = self._output_manager.step(controls | disturbances)
                reward, output_constraints_met = self._reward_manager.step(
                    controls | disturbances, outputs, control_constraints_met)
                log_variables = {
                    "Performance-Metrics": {
                        "Reward": reward,
                        "Control-Constraints-Met": int(control_constraints_met),
                        "Output-Constraints-Met": int(output_constraints_met)},
                    "Actions": actions,
                    "Controls": controls,
                    "Disturbances": disturbances,
                    "Outputs": outputs
                }
                self._experiment_tracker.step(log_variables, self._step_index)

                # TODO: use the lists used_outputs and used_controls to create the observations array,
                #  because unlike dicts, they are ordered
                observations = np.array(tuple(outputs.values()), dtype=np.float32)
                info = dict()
                self._step_index += 1

            except Exception as e:
                self.shutdown()
                raise e

        else:
            raise Exception("Cannot call step() before calling reset().")

        return observations, reward, False, False, info

    def reset(self) -> tuple[np.array, dict]:
        logger.info("Resetting environment...")
        try:
            self._step_index = 0
            self._config = self._initial_config.copy()
            self._disturbance_manager.reset()
            self._scenario_manager.reset()
            # scenario manager is capable of changing disturbances.
            disturbances = self._disturbance_manager.step()
            controls = self._control_manager.reset(disturbances)
            outputs = self._output_manager.reset(controls | disturbances)
            reward, output_constraints_met = self._reward_manager.reset(
                controls | disturbances, outputs, True)
            log_variables = {
                "Performance-Metrics": {
                    "Reward": reward,
                    "Control-Constraints-Met": 1,
                    "Output-Constraints-Met": int(output_constraints_met)},
                "Actions": {},
                "Controls": controls,
                "Disturbances": disturbances,
                "Outputs": outputs
            }
            self._experiment_tracker.reset(log_variables, self._step_index)

            # TODO: use the lists used_outputs and used_controls to create the observations array,
            #  because unlike dicts, they are ordered
            observations = np.array(tuple(outputs.values()), dtype=np.float32)

        except Exception as e:
            self.shutdown()
            raise e

        info = dict()
        self._ready = True
        logger.info("...environment has been reset.")

        return observations, info

    def shutdown(self) -> None:
        logger.info("Shutting down environment...")
        self._output_manager.shutdown()
        self._experiment_tracker.shutdown()
        logger.info("...environment has been shut down.")
