from omegaconf import DictConfig, OmegaConf
import importlib
import pathlib as pl
import logging
import sys

from adanowo_simulator.abstract_base_classes.control_manager import AbstractControlManager
from adanowo_simulator.calculation_adapter import CalculationAdapter

logger = logging.getLogger(__name__)

DEFAULT_RELATIVE_PATH = "./dependent_variable_calculations"


class ControlManager(AbstractControlManager):

    def __init__(self, config: DictConfig, actions_are_relative: bool = True):
        # use default path
        main_script_path = pl.Path(__file__).resolve().parent
        self._path_to_dependent_variable_calculations = main_script_path.parent / DEFAULT_RELATIVE_PATH

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
        self._config: DictConfig = OmegaConf.create()
        self._actions_are_relative: bool = actions_are_relative
        self._controls: dict[str, float] = dict()
        self._dependent_variable_calculations: dict[str, CalculationAdapter] = dict()
        self._ready: bool = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, actions: dict[str, float], disturbances: dict[str, float] | None) -> \
            tuple[dict[str, float], dict[str, float],  dict[str, bool], dict[str, bool]]:
        if disturbances is None:
            disturbances = dict()
        if self._ready:
            potential_controls = self._calculate_potential_controls(actions)
            potential_dependent_variables = \
                self._calculate_potential_dependent_variables(potential_controls | disturbances)
            control_constraints_met, dependent_variable_constraints_met = \
                self._constraints_met(potential_controls, potential_dependent_variables)

            if all((control_constraints_met | dependent_variable_constraints_met).values()):
                self._controls = potential_controls
            dependent_variables = \
                self._calculate_potential_dependent_variables(self._controls | disturbances)
        else:
            raise Exception("Cannot call step() before calling reset().")

        return self._controls, dependent_variables, control_constraints_met, dependent_variable_constraints_met

    def reset(self, disturbances: dict[str, float] | None) -> \
            tuple[dict[str, float], dict[str, float],  dict[str, bool], dict[str, bool]]:
        if disturbances is None:
            disturbances = dict()
        self._config = self._initial_config.copy()

        potential_controls = OmegaConf.to_container(self._config.initial_controls)
        if not self._dependent_variable_calculations:
            self._allocate_dependent_variable_calculations()
        potential_dependent_variables = \
            self._calculate_potential_dependent_variables(potential_controls | disturbances)
        control_constraints_met, dependent_variable_constraints_met = \
            self._constraints_met(potential_controls, potential_dependent_variables)

        if not all((control_constraints_met | dependent_variable_constraints_met).values()):
            raise AssertionError("The initial controls and dependent variables do not meet constraints. Aborting Experiment.")
        self._controls = potential_controls
        dependent_variables = \
            self._calculate_potential_dependent_variables(self._controls | disturbances)
        self._ready = True

        return self._controls, dependent_variables, control_constraints_met, dependent_variable_constraints_met

    def _constraints_met(self, controls: dict[str, float], dependent_variables: dict[str, float]) -> \
            tuple[dict[str, bool], dict[str, bool]]:
        # controls
        control_constraints_met = dict()
        for control_name, bounds in self._config.control_bounds.items():
            if "lower" in bounds.keys():
                if controls[control_name] < bounds.lower:
                    control_constraints_met[f"{control_name}.lower"] = False
                else:
                    control_constraints_met[f"{control_name}.lower"] = True
            if "upper" in bounds.keys():
                if controls[control_name] > bounds.upper:
                    control_constraints_met[f"{control_name}.upper"] = False
                else:
                    control_constraints_met[f"{control_name}.upper"] = True

        # dependent_variables
        dependent_variable_constraints_met = dict()
        for control_name, bounds in self._config.dependent_variable_bounds.items():
            if "lower" in bounds.keys():
                if dependent_variables[control_name] < bounds.lower:
                    dependent_variable_constraints_met[f"{control_name}.lower"] = False
                else:
                    dependent_variable_constraints_met[f"{control_name}.lower"] = True
            if "upper" in bounds.keys():
                if dependent_variables[control_name] > bounds.upper:
                    dependent_variable_constraints_met[f"{control_name}.upper"] = False
                else:
                    dependent_variable_constraints_met[f"{control_name}.upper"] = True

        return control_constraints_met, dependent_variable_constraints_met

    def _allocate_dependent_variable_calculations(self) -> None:
        for dependent_variable_name, calculation in self._config.dependent_variable_calculations.items():
            importlib.import_module(calculation)
            calculation_module = sys.modules[calculation]
            self._dependent_variable_calculations[dependent_variable_name] = CalculationAdapter(calculation_module)
            logger.info(f"Allocated calculation {calculation} to dependent variable {dependent_variable_name}.")

    def _calculate_potential_controls(self, actions: dict[str, float]) -> dict[str, float]:
        # relative actions.
        if self._config.actions_are_relative:
            potential_controls = dict()
            for control_name in self._config.initial_controls.keys():
                potential_controls[control_name] = self._controls[control_name] + actions[control_name]
        # absolute actions.
        else:
            potential_controls = actions

        return potential_controls

    def _calculate_potential_dependent_variables(self, X: dict[str, float]) -> dict[str, float]:
        potential_dependent_variables = dict()
        for dependent_variable_name, calculation in self._dependent_variable_calculations.items():
            potential_dependent_variables[dependent_variable_name] = calculation.calculate(X).item()
        return potential_dependent_variables

