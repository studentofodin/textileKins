import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.state_manager import AbstractStateManager


class StateManager(AbstractStateManager):

    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._action_names = [name + '_action' for name in list(config.initial_controls.keys())]
        self._n_controls = len(config.initial_controls)
        self._n_disturbances = len(config.disturbances)
        self._config = None
        self._controls = None
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def n_disturbances(self) -> int:
        return self._n_disturbances

    def get_state(self, action: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:
        # relative actions.
        if self._config.actions_are_relative:
            updated_controls = dict()
            for index, control in enumerate(self._controls.keys()):
                updated_controls[control] = self._controls[control] + action[index]
        # absolute actions.
        else:
            updated_controls = dict(zip(self._controls.keys(), action.tolist()))

        control_constraints_met = self._control_constraints_met(updated_controls)
        if control_constraints_met:
            self._controls = updated_controls

        state = self._controls | dict(self._config.disturbances)

        action_dict = dict(zip(self._action_names, action.tolist()))

        return state, control_constraints_met, action_dict

    def reset(self) -> dict[str, float]:
        self._config = self._initial_config.copy()
        self._controls = dict(self._config.initial_controls)
        if not self._control_constraints_met(self._controls):
            raise AssertionError("The initial setting does not meet control constraints. Aborting Experiment.")
        state = self._controls | dict(self._config.disturbances)
        return state

    def _control_constraints_met(self, controls: dict[str, float]) -> bool:
        # assume that safety constraints are met.
        control_constraints_met = True

        # check simple fixed bounds for controls.
        for control_name, bounds in self._config.control_bounds.items():
            if (controls[control_name] < bounds.lower) or (controls[control_name] > bounds.upper):
                control_constraints_met = False

        return control_constraints_met
