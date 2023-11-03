import sys
import logging
from copy import deepcopy
from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.environment import AbstractEnvironment
from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager
from adanowo_simulator.abstract_base_classes.objective_manager import AbstractObjectiveManager
from adanowo_simulator.abstract_base_classes.action_manager import AbstractActionManager
from adanowo_simulator.abstract_base_classes.disturbance_manager import AbstractDisturbanceManager
from adanowo_simulator.abstract_base_classes.experiment_tracker import AbstractExperimentTracker
from adanowo_simulator.abstract_base_classes.scenario_manager import AbstractScenarioManager

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(AbstractEnvironment):
    def __init__(
            self, config: DictConfig, disturbance_manager: AbstractDisturbanceManager,
            action_manager: AbstractActionManager, output_manager: AbstractOutputManager,
            objective_manager: AbstractObjectiveManager, scenario_manager: AbstractScenarioManager,
            experiment_tracker: AbstractExperimentTracker):
        self._disturbance_manager: AbstractDisturbanceManager = disturbance_manager
        self._action_manager: AbstractActionManager = action_manager
        self._output_manager: AbstractOutputManager = output_manager
        self._objective_manager: AbstractObjectiveManager = objective_manager
        self._scenario_manager: AbstractScenarioManager = scenario_manager
        self._experiment_tracker: AbstractExperimentTracker = experiment_tracker

        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = self._initial_config.copy()
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
    def disturbance_manager(self) -> AbstractDisturbanceManager:
        return self._disturbance_manager

    @property
    def action_manager(self) -> AbstractActionManager:
        return self._action_manager

    @property
    def output_manager(self) -> AbstractOutputManager:
        return self._output_manager

    @property
    def objective_manager(self) -> AbstractObjectiveManager:
        return self._objective_manager

    @property
    def scenario_manager(self) -> AbstractScenarioManager:
        return self._scenario_manager

    @property
    def experiment_tracker(self) -> AbstractExperimentTracker:
        return self._experiment_tracker

    @property
    def step_index(self):
        return self._step_index

    def step(self, actions: dict) -> tuple[float, dict[str, float], dict[str, float], DictConfig]:
        if self._ready:
            try:
                if self._step_index == 1:
                    logger.info("Experiment is running.")
                disturbances = self._disturbance_manager.step()
                setpoints, dependent_variables, setpoints_okay, dependent_variables_okay = \
                    self._action_manager.step(actions, disturbances)
                state = disturbances | setpoints | dependent_variables
                outputs = self._output_manager.step(state)
                objective_value, output_constraints_met = self._objective_manager.step(
                    state, outputs, setpoints_okay, dependent_variables_okay)
                log_variables = {
                    "Performance-Metrics": {
                        "Objective-Value": objective_value,
                        "Setpoint-Constraints-Met": int(all(setpoints_okay.values())),
                        "Dependent-Variable-Constraints-Met": int(all(dependent_variables_okay.values())),
                        "Output-Constraints-Met": int(all(output_constraints_met.values()))
                    },
                    "Actions": actions,
                    "Output-Constraints-Met": {key: int(value) for key, value in output_constraints_met.items()},
                    "Outputs": outputs,
                    "Setpoint-Constraints-Met": {key: int(value) for key, value in setpoints_okay.items()},
                    "Setpoints": setpoints,
                    "Dependent-Variable-Constraints-Met":
                        {key: int(value) for key, value in dependent_variables_okay.items()},
                    "Dependent-Variables": dependent_variables,
                    "Disturbances": disturbances
                }
                self._experiment_tracker.step(log_variables, self._step_index)

                # Prepare next step.
                # Execute scenario for the next step so the agent is already informed about production context changes.
                self._step_index += 1
                self._scenario_manager.step(self._step_index, self._disturbance_manager, self._output_manager,
                                            self._objective_manager)
                disturbances = self._disturbance_manager.step()
                state = disturbances | setpoints | dependent_variables
                quality_bounds = deepcopy(self._objective_manager.config.output_bounds)

            except Exception as e:
                self.close()
                raise e

        else:
            raise Exception("Cannot call step() before calling reset().")

        return objective_value, state, outputs, quality_bounds

    def reset(self) -> tuple[float, dict[str, float], dict[str, float], DictConfig]:
        logger.info("Resetting environment...")
        try:
            # step 0.
            self._step_index = 0
            self._config = self._initial_config.copy()
            self._scenario_manager.reset()
            disturbances = self._disturbance_manager.reset()
            setpoints, dependent_variables, setpoint_constraints_met, dependent_variable_constraints_met = \
                self._action_manager.reset(disturbances)
            state = disturbances | setpoints | dependent_variables
            outputs = self._output_manager.reset(state)
            objective_value, output_constraints_met = self._objective_manager.reset(
                state, outputs, setpoint_constraints_met, dependent_variable_constraints_met)
            log_variables = {
                "Performance-Metrics": {
                    "Objective-Value": objective_value,
                    "Setpoint-Constraints-Met": int(all(setpoint_constraints_met.values())),
                    "Dependent-Variable-Constraints-Met": int(all(dependent_variable_constraints_met.values())),
                    "Output-Constraints-Met": int(all(output_constraints_met.values()))
                },
                "Actions": {},
                "Output-Constraints-Met": {key: int(value) for key, value in output_constraints_met.items()},
                "Outputs": outputs,
                "Setpoint-Constraints-Met": {key: int(value) for key, value in setpoint_constraints_met.items()},
                "Setpoints": setpoints,
                "Dependent-Variable-Constraints-Met":
                    {key: int(value) for key, value in dependent_variable_constraints_met.items()},
                "Dependent-Variables": dependent_variables,
                "Disturbances": disturbances
            }
            self._experiment_tracker.reset(log_variables)

            # prepare step 1.
            self._step_index = 1
            self._scenario_manager.step(self._step_index, self._disturbance_manager, self._output_manager,
                                        self._objective_manager)
            disturbances = self._disturbance_manager.step()
            state = disturbances | setpoints | dependent_variables
            quality_bounds = deepcopy(self._objective_manager.config.output_bounds)

        except Exception as e:
            self.close()
            raise e

        self._ready = True
        logger.info("...environment has been reset.")

        return objective_value, state, outputs, quality_bounds

    def close(self) -> None:
        logger.info("Closing environment...")
        exceptions = []
        try:
            self._disturbance_manager.close()
        except Exception as e:
            exceptions.append(e)
        try:
            self._action_manager.close()
        except Exception as e:
            exceptions.append(e)
        try:
            self._output_manager.close()
        except Exception as e:
            exceptions.append(e)
        try:
            self._objective_manager.close()
        except Exception as e:
            exceptions.append(e)
        try:
            self._scenario_manager.close()
        except Exception as e:
            exceptions.append(e)
        try:
            self._experiment_tracker.close()
        except Exception as e:
            exceptions.append(e)
        self._step_index = -1
        self._ready = False
        if exceptions:
            raise Exception(exceptions)
        logger.info("...environment has been closed.")
