from typing import Any

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
    def __init__(
            self, config: DictConfig, disturbance_manager: AbstractDisturbanceManager,
            control_manager: AbstractControlManager, output_manager: AbstractOutputManager,
            reward_manager: AbstractRewardManager, scenario_manager: AbstractScenarioManager,
            experiment_tracker: AbstractExperimentTracker):
        self._disturbance_manager: AbstractDisturbanceManager = disturbance_manager
        self._control_manager: AbstractControlManager = control_manager
        self._output_manager: AbstractOutputManager = output_manager
        self._reward_manager: AbstractRewardManager = reward_manager
        self._scenario_manager: AbstractScenarioManager = scenario_manager
        self._experiment_tracker: AbstractExperimentTracker = experiment_tracker

        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = OmegaConf.create()
        self._step_index: int = -1
        self._ready: bool = False
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
    def step_index(self):
        return self._step_index

    def _array_to_dict(self, array: np.array, keys: list[str]) -> dict[str, float]:
        if array.size != len(keys):
            raise Exception("Length of array and keys are not the same.")
        dictionary = dict()
        for index, key in enumerate(keys):
            dictionary[key] = array[index]
        return dictionary

    @staticmethod
    def _dict_to_array(dictionary: dict[str, float], keys: list[str]) -> np.array:
        if len(dictionary) != len(keys):
            raise Exception("Length of dictionary and keys are not the same.")
        array = np.zeros(len(dictionary))
        for index, key in enumerate(keys):
            array[index] = dictionary[key]
        return array

    def _collect_observations(self, state: dict[str, float], outputs: dict[str, float]) -> np.array:
        # fill observations
        observations = list()
        for component_name in self._config.observations:
            if component_name == "disturbances":
                for disturbance_name in self._config.used_disturbances:
                    observations.append(state[disturbance_name])
            elif component_name == "controls":
                for control_name in self._config.used_controls:
                    observations.append(state[control_name])
            elif component_name == "dependent_variables":
                for dependent_variable_name in self._config.used_dependent_variables:
                    observations.append(state[dependent_variable_name])
            elif component_name == "outputs":
                for output_name in self._config.used_outputs:
                    observations.append(outputs[output_name])
            else:
                raise Exception(f"{component_name} in observation config is not known!")
        observations = np.array(observations)
        return observations

    def step(self, actions_array: np.array) -> tuple[Any, float]:
        if self._ready:
            try:
                if self._step_index == 1:
                    logger.info("Experiment is running.")
                actions = self._array_to_dict(actions_array, OmegaConf.to_container(self._config.used_controls))
                disturbances = self._disturbance_manager.step()
                controls, dependent_variables, control_constraints_met, dependent_variable_constraints_met = \
                    self._control_manager.step(actions, disturbances)
                state = disturbances | controls | dependent_variables
                outputs = self._output_manager.step(state)
                reward, output_constraints_met = self._reward_manager.step(
                    state, outputs, control_constraints_met, dependent_variable_constraints_met)
                log_variables = {
                    "Performance-Metrics": {
                        "Reward": reward,
                        "Control-Constraints-Met": int(all(control_constraints_met.values())),
                        "Dependent-Variable-Constraints-Met": int(all(dependent_variable_constraints_met.values())),
                        "Output-Constraints-Met": int(all(output_constraints_met.values()))},
                    "Control-Constraints-Met": {key: int(value) for key, value in control_constraints_met.items()},
                    "Dependent-Variable-Constraints-Met":
                        {key: int(value) for key, value in dependent_variable_constraints_met.items()},
                    "Output-Constraints-Met": {key: int(value) for key, value in output_constraints_met.items()},
                    "Actions": actions,
                    "Controls": controls,
                    "Disturbances": disturbances,
                    "Outputs": outputs}
                self._experiment_tracker.step(log_variables, self._step_index)

                # prepare next step.
                self._step_index += 1
                self._scenario_manager.step(self._step_index, self._disturbance_manager, self._output_manager,
                                            self._reward_manager)
                disturbances = self._disturbance_manager.step()
                state = disturbances | controls | dependent_variables
                observations = self._collect_observations(state, outputs)

            except Exception as e:
                self.close()
                raise e

        else:
            raise Exception("Cannot call step() before calling reset().")

        return observations, reward

    def reset(self) -> tuple[Any, float]:
        logger.info("Resetting environment...")
        try:
            # step 0.
            self._step_index = 0
            self._config = self._initial_config.copy()
            self._scenario_manager.reset()
            disturbances = self._disturbance_manager.reset()
            controls, dependent_variables, control_constraints_met, dependent_variable_constraints_met = \
                self._control_manager.reset(disturbances)
            state = disturbances | controls | dependent_variables
            outputs = self._output_manager.reset(state)
            reward, output_constraints_met = self._reward_manager.reset(
                state, outputs, control_constraints_met, dependent_variable_constraints_met)
            log_variables = {
                "Performance-Metrics": {
                    "Reward": reward,
                    "Control-Constraints-Met": int(all(control_constraints_met.values())),
                    "Dependent-Variable-Constraints-Met": int(all(dependent_variable_constraints_met.values())),
                    "Output-Constraints-Met": int(all(output_constraints_met.values()))},
                "Control-Constraints-Met": {key: int(value) for key, value in control_constraints_met.items()},
                "Dependent-Variable-Constraints-Met":
                    {key: int(value) for key, value in dependent_variable_constraints_met.items()},
                "Output-Constraints-Met": {key: int(value) for key, value in output_constraints_met.items()},
                "Actions": {},
                "Controls": controls,
                "Disturbances": disturbances,
                "Outputs": outputs}
            self._experiment_tracker.reset(log_variables, self._step_index)

            # prepare step 1.
            self._step_index = 1
            self._scenario_manager.step(self._step_index, self._disturbance_manager, self._output_manager,
                                        self._reward_manager)
            disturbances = self._disturbance_manager.step()
            state = disturbances | controls | dependent_variables
            observations = self._collect_observations(state, outputs)

        except Exception as e:
            self.close()
            raise e

        self._ready = True
        logger.info("...environment has been reset.")

        return observations, reward

    def close(self) -> None:
        logger.info("Closing environment...")
        self._output_manager.close()
        self._experiment_tracker.close()
        self._config = OmegaConf.create()
        self._step_index = -1
        self._ready = False
        logger.info("...environment has been closed.")
