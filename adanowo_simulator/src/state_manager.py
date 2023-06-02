import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.state_manager import AbstractStateManager


class StateManager(AbstractStateManager):

    def __init__(self, config: DictConfig, relative_actions: bool):
        self._relative_actions = relative_actions
        self._initial_config = config.copy()
        self._config = config.copy()
        self._action_names = [name + '_action' for name in list(config.initial_controls.keys())]
        self._n_controls = len(config.initial_controls)
        self._n_disturbances = len(config.disturbances)
        self._controls = dict(self._config.initial_controls)

        if not self._check_safety(self._controls):
            raise AssertionError("The initial setting is unsafe. Aborting Experiment.")

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def action_type(self) -> bool:
        return self._relative_actions

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def n_disturbances(self) -> int:
        return self._n_disturbances

    def _calc_new_controls(self, action: np.array) -> dict[str, float]:
        if self._relative_actions:
            updated_controls = dict()
            for index, control in enumerate(self._controls.keys()):
                updated_controls[control] = self._controls[control] + action[index]
        else:
            updated_controls = dict(zip(self._controls.keys(), action.tolist()))

        return updated_controls

    def get_state(self) -> dict[str, float]:
        state = self._controls | dict(self._config.disturbances)
        return state

    def update_state(self, action: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:

        updated_controls = self._calc_new_controls(action)

        safety_met = self._check_safety(updated_controls)
        if safety_met:
            self._controls = updated_controls

        action_dict = dict(zip(self._action_names, action.tolist()))

        return self.get_state(), safety_met, action_dict

    def reset(self) -> dict[str, float]:
        self._config = self._initial_config.copy()
        self._controls = dict(self._initial_config.initial_controls)
        return self.get_state()

    def _check_safety(self, controls: dict[str, float]) -> bool:
        # assume that safety constraints are met.
        safety_met = True

        # check simple fixed bounds for controls.
        for control_name, bounds in self._config.safety.simple_control_bounds.items():
            if (controls[control_name] < bounds.lower) or (controls[control_name] > bounds.upper):
                safety_met = False

        return safety_met
