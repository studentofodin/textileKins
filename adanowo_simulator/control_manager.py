from omegaconf import DictConfig, OmegaConf
import importlib
import pathlib as pl
import logging
import sys

from adanowo_simulator.abstract_base_classes.control_manager import AbstractControlManager
from adanowo_simulator.calculation_adapter import CalculationAdapter

logger = logging.getLogger(__name__)

DEFAULT_RELATIVE_PATH = "./secondary_control_calculations"


class ControlManager(AbstractControlManager):

    def __init__(self, config: DictConfig, actions_are_relative: bool = True):
        # use default path
        main_script_path = pl.Path(__file__).resolve().parent
        self._path_to_secondary_control_calculations = main_script_path.parent / DEFAULT_RELATIVE_PATH

        if config.path_to_secondary_control_calculations is not None:
            temp_path = pl.Path(config.path_to_secondary_control_calculations)
            if self._path_to_secondary_control_calculations.is_dir():
                logger.info(f"Using custom secondary control calculation path "
                            f"{self._path_to_secondary_control_calculations}.")
                self._path_to_secondary_control_calculations = temp_path
            else:
                raise Exception(
                    f"Custom secondary control calculation path {self._path_to_secondary_control_calculations}"
                    f" is not valid.")

        # Add calculation path to sys.path so that the calculations can be imported.
        sys.path.append(str(self._path_to_secondary_control_calculations))

        self._initial_config = config.copy()
        self._config = None
        self._actions_are_relative = actions_are_relative
        self._controls = None
        self._secondary_control_calculations = dict()
        self._ready = False

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, actions: dict[str, float], disturbances: dict[str, float] = dict()) -> tuple[dict[str, float], bool]:
        if self._ready:
            potential_primary_controls = self._calculate_potential_primary_controls(actions)
            X = potential_primary_controls | disturbances
            potential_secondary_controls = self._calculate_potential_secondary_controls(X)
            potential_controls = potential_primary_controls | potential_secondary_controls

            control_constraints_met = self._control_constraints_met(potential_controls)
            if control_constraints_met:
                self._controls = potential_controls
        else:
            raise Exception("Cannot call step() before calling reset().")

        return self._controls, control_constraints_met

    def reset(self, disturbances: dict[str, float] = dict()) -> dict[str, float]:
        self._config = self._initial_config.copy()

        potential_primary_controls = OmegaConf.to_container(self._config.initial_primary_controls)
        if not self._secondary_control_calculations:
            self._allocate_secondary_control_calculations()
        X = potential_primary_controls | disturbances
        potential_secondary_controls = self._calculate_potential_secondary_controls(X)
        potential_controls = potential_primary_controls | potential_secondary_controls

        if self._control_constraints_met(potential_controls):
            self._controls = potential_controls
        else:
            raise AssertionError("The initial controls do not meet control constraints. Aborting Experiment.")
        self._ready = True
        return self._controls

    def _control_constraints_met(self, controls: dict[str, float]) -> bool:
        # assume that control constraints are met.
        control_constraints_met = True

        for control_name, bounds in self._config.control_bounds.items():
            if "lower" in bounds.keys():
                if controls[control_name] < bounds.lower:
                    control_constraints_met = False
            if "upper" in bounds.keys():
                if controls[control_name] > bounds.upper:
                    control_constraints_met = False

        return control_constraints_met

    def _allocate_secondary_control_calculations(self) -> None:
        for control_name, calculation_name in self._config.secondary_control_calculations.items():
            importlib.import_module(calculation_name)
            calculation_module = sys.modules[calculation_name]
            self._secondary_control_calculations[control_name] = CalculationAdapter(calculation_module)
            logger.info(f"Allocated calculation {calculation_name} to secondary control {control_name}.")

    def _calculate_potential_primary_controls(self, actions: dict[str, float]) -> dict[str, float]:
        # relative actions.
        if self._config.actions_are_relative:
            potential_primary_controls = dict()
            for control_name in self._config.initial_primary_controls.keys():
                potential_primary_controls[control_name] = self._controls[control_name] + actions[control_name]
        # absolute actions.
        else:
            potential_primary_controls = actions

        return potential_primary_controls

    def _calculate_potential_secondary_controls(self, X: dict[str, float]) -> dict[str, float]:
        potential_secondary_controls = dict()
        for control_name, calculation in self._secondary_control_calculations.items():
            potential_secondary_controls[control_name] = calculation.calculate(X)
        return potential_secondary_controls

