import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.state_manager import AbstractStateManager


class StateManager(AbstractStateManager):

    def __init__(self, config: DictConfig, action_type: int):
        self._action_type = action_type
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
    def action_type(self) -> int:
        return self._action_type

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def n_disturbances(self) -> int:
        return self._n_disturbances

    def get_state(self, action: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:
        # relative actions.
        if self._action_type == 0:
            updated_controls = dict()
            for index, control in enumerate(self._controls.keys()):
                updated_controls[control] = self._controls[control] + action[index]
        # absolute actions.
        else:
            updated_controls = dict(zip(self._controls.keys(), action.tolist()))

        safety_met = self._safety_met(updated_controls)
        if safety_met:
            self._controls = updated_controls

        state = self._controls | dict(self._config.disturbances)

        action_dict = dict(zip(self._action_names, action.tolist()))

        return state, safety_met, action_dict

    def reset(self) -> dict[str, float]:
        self._config = self._initial_config.copy()
        self._controls = dict(self._config.initial_controls)
        if not self._safety_met(self._controls):
            raise AssertionError("The initial setting is unsafe. Aborting Experiment.")
        state = self._controls | dict(self._config.disturbances)
        return state

    def _safety_met(self, controls: dict[str, float]) -> bool:
        # assume that safety constraints are met.
        safety_met = True

        # check simple fixed bounds for controls.
        for control_name, bounds in self._config.safety.simple_control_bounds.items():
            if (controls[control_name] < bounds.lower) or (controls[control_name] > bounds.upper):
                safety_met = False

        # check more complex, relational constraints.
        # constr_sum = 0
        # for control_name, value in controls.items():
        #     constr_sum = constr_sum + value
        # if (constr_sum < self._config.safety.complex_constraints.add_min) or \
        #         (constr_sum > self._config.safety.complex_constraints.add_max):
        #     safety_met = False

        return safety_met
