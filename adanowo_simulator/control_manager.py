from omegaconf import DictConfig, OmegaConf
import importlib
import pathlib as pl
import logging

from adanowo_simulator.abstract_base_classes.control_manager import AbstractControlManager

logger = logging.getLogger(__name__)


class ControlManager(AbstractControlManager):

    def __init__(self, config: DictConfig, actions_are_relative: bool = True):
        self._initial_config = config.copy()
        self._config = None
        self._actions_are_relative = actions_are_relative
        self._controls = None
        self._secondary_control_models = dict()
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
        if not self._secondary_control_models:
            self._allocate_secondary_control_models()
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

    def _allocate_secondary_control_models(self) -> None:
        for control_name, model_name in self._config.secondary_control_models.items():
            model_path = pl.Path(self._config.path_to_secondary_control_models) / f"{model_name}.py"
            try:
                spec = importlib.util.spec_from_file_location("module.name", model_path)
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                self._secondary_control_models[control_name] = model_module.SecondaryControlModel
                logger.info(f"Allocated model {model_name} to secondary control {control_name}.")
            except Exception as e:
                logger.error(f"Could not allocate model {model_name} to secondary control {control_name}.")
                raise e

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
        for control_name, model in self._secondary_control_models.items():
            potential_secondary_controls[control_name] = model.calculate_control(X)
        return potential_secondary_controls

