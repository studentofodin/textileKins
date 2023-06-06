import numpy as np
from omegaconf import DictConfig, OmegaConf


from src.abstract_base_class.control_manager import AbstractControlManager


class ControlManager(AbstractControlManager):

    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self._action_names = [name + '_action' for name in list(config.initial_controls.keys())]
        self._n_controls = len(config.initial_controls)
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

    def step(self, actions: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:
        # relative actions.
        if self._config.actions_are_relative:
            potential_controls = dict()
            for index, control in enumerate(self._controls.keys()):
                potential_controls[control] = self._controls[control] + actions[index]
        # absolute actions.
        else:
            potential_controls = dict(zip(self._controls.keys(), actions.tolist()))

        control_constraints_met = self._control_constraints_met(potential_controls)
        if control_constraints_met:
            self._controls = potential_controls

        actions_dict = dict(zip(self._action_names, actions.tolist()))

        return self._controls, control_constraints_met, actions_dict

    def reset(self) -> dict[str, float]:
        self._config = self._initial_config.copy()
        self._controls = OmegaConf.to_container(self._config.initial_controls)
        if not self._control_constraints_met(self._controls):
            raise AssertionError("The initial controls do not meet safety constraints. Aborting Experiment.")
        return self._controls

    def _control_constraints_met(self, controls: dict[str, float]) -> bool:
        # assume that safety constraints are met.
        control_constraints_met = True

        # check simple fixed bounds for controls.
        for control_name, bounds in self._config.control_bounds.items():
            if (controls[control_name] < bounds.lower) or (controls[control_name] > bounds.upper):
                control_constraints_met = False

        return control_constraints_met
