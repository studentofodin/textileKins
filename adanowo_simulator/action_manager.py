from omegaconf import DictConfig, OmegaConf
import importlib
import pathlib as pl
import logging
import sys

from adanowo_simulator.abstract_base_classes.action_manager import AbstractActionManager
from adanowo_simulator.calculation_adapter import CalculationAdapter

logger = logging.getLogger(__name__)

DEFAULT_RELATIVE_PATH = "dependent_variable_calculations"


class ActionManager(AbstractActionManager):

    def __init__(self, config: DictConfig, actions_are_relative: bool = True):
        # use default path
        script_path = pl.Path(__file__).resolve().parent
        self._path_to_dependent_variable_calculations = script_path / DEFAULT_RELATIVE_PATH

        if config.path_to_dependent_variable_calculations is not None:
            temp_path = pl.Path(config.path_to_dependent_variable_calculations)
            if self._path_to_dependent_variable_calculations.is_dir():
                logger.info(f"Using custom dependent variable calculation path "
                            f"{self._path_to_dependent_variable_calculations}.")
                self._path_to_dependent_variable_calculations = temp_path
            else:
                raise Exception(
                    f"Custom dependent variable calculation path {self._path_to_dependent_variable_calculations}"
                    f" is not valid.")

        # Add calculation path to sys.path so that the calculations can be imported.
        sys.path.append(str(self._path_to_dependent_variable_calculations))

        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = self._initial_config.copy()
        self._actions_are_relative: bool = actions_are_relative
        self._setpoints: dict[str, float] = dict()
        self._dependent_variable_calculations: dict[str, CalculationAdapter] = dict()
        self._ready: bool = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, actions: dict[str, float], disturbances: dict[str, float]) -> \
            tuple[dict[str, float], dict[str, float],  dict[str, bool], dict[str, bool]]:
        if self._ready:
            potential_setpoints = self._calculate_potential_setpoints(actions)
            potential_dependent_variables = \
                self._calculate_potential_dependent_variables(potential_setpoints | disturbances)
            setpoint_constraints_met, dependent_variable_constraints_met = \
                self._constraints_met(potential_setpoints, potential_dependent_variables)

            if all((setpoint_constraints_met | dependent_variable_constraints_met).values()):
                self._setpoints = potential_setpoints
            dependent_variables = \
                self._calculate_potential_dependent_variables(self._setpoints | disturbances)
        else:
            raise Exception("Cannot call step() before calling reset().")

        return self._setpoints, dependent_variables, setpoint_constraints_met, dependent_variable_constraints_met

    def reset(self, initial_disturbances: dict[str, float]) -> \
            tuple[dict[str, float], dict[str, float],  dict[str, bool], dict[str, bool]]:
        self._config = self._initial_config.copy()

        potential_setpoints = OmegaConf.to_container(self._config.initial_setpoints)
        if not self._dependent_variable_calculations:
            self._allocate_dependent_variable_calculations()
        potential_dependent_variables = \
            self._calculate_potential_dependent_variables(potential_setpoints | initial_disturbances)
        setpoint_constraints_met, dependent_variable_constraints_met = \
            self._constraints_met(potential_setpoints, potential_dependent_variables)

        if not all((setpoint_constraints_met | dependent_variable_constraints_met).values()):
            raise AssertionError("The initial setpoints and dependent variables do not meet constraints. "
                                 "Aborting Experiment.")
        self._setpoints = potential_setpoints
        dependent_variables = \
            self._calculate_potential_dependent_variables(self._setpoints | initial_disturbances)
        self._ready = True

        return self._setpoints, dependent_variables, setpoint_constraints_met, dependent_variable_constraints_met

    def close(self) -> None:
        self._ready = False

    def _constraints_met(self, setpoints: dict[str, float], dependent_variables: dict[str, float]) -> \
            tuple[dict[str, bool], dict[str, bool]]:

        def check_constraints(boundaries_to_check: DictConfig, actual_vars: dict[str, float]):
            constraints_met = dict()
            for ctrl_name, boundaries in boundaries_to_check.items():
                for boundary_type in ["lower", "upper"]:
                    boundary_value = boundaries.get(boundary_type)
                    if boundary_value is not None:
                        comparison = actual_vars[ctrl_name] < boundary_value if boundary_type == "lower" else \
                            actual_vars[ctrl_name] > boundary_value
                        constraints_met[f"{ctrl_name}.{boundary_type}"] = not comparison

            return constraints_met

        setpoint_constraints_met = check_constraints(self._config.setpoint_bounds, setpoints)
        dependent_variable_constraints_met = check_constraints(self._config.dependent_variable_bounds,
                                                               dependent_variables)

        return setpoint_constraints_met, dependent_variable_constraints_met

    def _allocate_dependent_variable_calculations(self) -> None:
        for dependent_variable_name, calculation in self._config.dependent_variable_calculations.items():
            importlib.import_module(calculation)
            calculation_module = sys.modules[calculation]
            self._dependent_variable_calculations[dependent_variable_name] = CalculationAdapter(calculation_module)
            logger.info(f"Allocated calculation {calculation} to dependent variable {dependent_variable_name}.")

    def _calculate_potential_setpoints(self, actions: dict[str, float]) -> dict[str, float]:
        # relative actions.
        if self._config.actions_are_relative:
            potential_setpoints = dict()
            for setpoint_name in self._config.initial_setpoints.keys():
                potential_setpoints[setpoint_name] = self._setpoints[setpoint_name] + actions[setpoint_name]
        # absolute actions.
        else:
            potential_setpoints = actions

        return potential_setpoints

    def _calculate_potential_dependent_variables(self, X: dict[str, float]) -> dict[str, float]:
        potential_dependent_variables = dict()
        for dependent_variable_name, calculation in self._dependent_variable_calculations.items():
            potential_dependent_variables[dependent_variable_name] = calculation.calculate(X).item()
        return potential_dependent_variables

