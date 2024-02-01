import sys
import logging
from copy import copy
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


def compile_log_variables(
    objective_value,
    setpoints_okay,
    dependent_variables_okay,
    output_constraints_met,
    actions,
    outputs,
    setpoints,
    dependent_variables,
    disturbances
) -> dict:
    log_dict = {
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
        "Dependent-Variable-Constraints-Met": {key: int(value) for key, value in dependent_variables_okay.items()},
        "Dependent-Variables": dependent_variables,
        "Disturbances": disturbances
    }
    return log_dict


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

        self.log_vars = None
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
        if not self._ready:
            raise RuntimeError("Cannot call step() before calling reset().")
        try:
            if self._step_index == 1:
                logger.info("Experiment is running.")
            assert set(actions.keys()) == set(self._config.used_setpoints), "Action dict does not match used setpoints."
            disturbances = self._disturbance_manager.step()
            setpoints, dependent_variables, setpoints_okay, dependent_variables_okay = \
                self._action_manager.step(actions, disturbances)
            state = disturbances | setpoints | dependent_variables
            outputs = self._output_manager.step(state)
            objective_value, output_constraints_met = self._objective_manager.step(
                state, outputs, setpoints_okay, dependent_variables_okay)
            log_variables = compile_log_variables(
                objective_value,
                setpoints_okay,
                dependent_variables_okay,
                output_constraints_met,
                actions,
                outputs,
                setpoints,
                dependent_variables,
                disturbances
            )
            self._experiment_tracker.step(log_variables, self._step_index)
            self.log_vars = copy(log_variables)

            # Execute scenario for the next step so the agent is already informed about production context changes.
            state_with_new_context, quality_bounds_next = self._prepare_next_step(setpoints, dependent_variables)

        except Exception as e:
            self.close()
            raise e

        return objective_value, state_with_new_context, outputs, quality_bounds_next

    def reset(self) -> tuple[float, dict[str, float], dict[str, float], DictConfig]:
        logger.info("Resetting environment...")
        try:
            # step 0.
            self._step_index = 0
            self._config = self._initial_config.copy()
            self._scenario_manager.reset()
            disturbances = self._disturbance_manager.reset()
            setpoints, dependent_variables, setpoints_okay, dependent_variables_okay = \
                self._action_manager.reset(disturbances)
            state = disturbances | setpoints | dependent_variables
            outputs = self._output_manager.reset(state)
            objective_value, output_constraints_met = self._objective_manager.reset(
                state, outputs, setpoints_okay, dependent_variables_okay)
            log_variables = compile_log_variables(
                objective_value,
                setpoints_okay,
                dependent_variables_okay,
                output_constraints_met,
                {},
                outputs,
                setpoints,
                dependent_variables,
                disturbances
            )
            self._experiment_tracker.reset(log_variables)
            self.log_vars = copy(log_variables)

            # prepare step 1.
            state_with_new_context, quality_bounds_next = self._prepare_next_step(setpoints, dependent_variables)

        except Exception as e:
            self.close()
            raise e

        self._ready = True
        logger.info("...environment has been reset.")

        return objective_value, state_with_new_context, outputs, quality_bounds_next

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

    def _prepare_next_step(self, setpoints, dependent_variables):
        self._step_index += 1
        self._scenario_manager.step(self._step_index, self._disturbance_manager, self._output_manager,
                                    self._objective_manager)
        disturbances = self._disturbance_manager.step()
        state_with_new_context = disturbances | setpoints | dependent_variables
        quality_bounds_next = copy(self._objective_manager.config.output_bounds)
        return state_with_new_context, quality_bounds_next
