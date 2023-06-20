import numpy as np
from omegaconf import DictConfig, OmegaConf


from adanowo_simulator.abstract_base_classes.control_manager import AbstractControlManager


class ControlManager(AbstractControlManager):

    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._config = config.copy()
        self._controls = dict(self._config.initial_controls)
        if not self._control_constraints_met(self._controls):
            raise AssertionError("The initial controls do not meet control constraints. Aborting Experiment.")

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, actions: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:

        actions_dict = dict(zip(self._controls.keys(), actions.tolist()))

        # relative actions.
        if self._config.actions_are_relative:
            potential_controls = dict()
            for control_name in self._controls.keys():
                potential_controls[control_name] = self._controls[control_name] + actions_dict[control_name]
        # absolute actions.
        else:
            potential_controls = actions_dict

        control_constraints_met = self._control_constraints_met(potential_controls)
        if control_constraints_met:
            self._controls = potential_controls

        return self._controls, control_constraints_met, actions_dict

    def reset(self) -> dict[str, float]:
        self._config = self._initial_config.copy()
        self._controls = OmegaConf.to_container(self._config.initial_controls)
        if not self._control_constraints_met(self._controls):
            raise AssertionError("The initial controls do not meet control constraints. Aborting Experiment.")
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
