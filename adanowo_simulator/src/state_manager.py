import numpy as np
from omegaconf import DictConfig

from src.abstract_base_class.state_manager import AbstractStateManager


class StateManager(AbstractStateManager):

    def __init__(self, config: DictConfig, actionType: int):
        self._initialConfig = config.copy()
        self._n_controls = len(config.initialControls)
        self._n_disturbances = len(config.disturbances)
        self._actionType = actionType
        self.reset()
        self._actionNames = [name+'_action' for name in list(self._controls.keys())]
        self._controls = None

        self._config = None

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def actionType(self) -> int:
        return self._actionType

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def n_disturbances(self) -> int:
        return self._n_disturbances

    def getState(self, action: np.array) -> tuple[dict[str, float], bool, dict[str, float]]:
        # relative actions.
        if self._actionType == 0:
            updatedControls = dict()
            for index, control in enumerate(self._controls.keys()):
                updatedControls[control] = self._controls[control] + action[index]
        # absolute actions.
        else:
            updatedControls = dict(zip(self._controls.keys(), action.tolist()))

        safetyMet = self._safetyMet(updatedControls)
        if safetyMet:
            self._controls = updatedControls

        state = self._controls | dict(self._config.disturbances)

        actionDict = dict(zip(self._actionNames, action.tolist()))

        return state, safetyMet, actionDict

    def reset(self) -> dict[str, float]:
        self._config = self._initialConfig.copy()
        self._controls = dict(self._config.initialControls)
        if not self._safetyMet(self._controls):
            raise AssertionError("The initial setting is unsafe. Aborting Experiment.")
        state = self._controls | dict(self._config.disturbances)
        return state

    def _safetyMet(self, controls: dict[str, float]) -> bool:
        # assume that safety constraints are met.
        safetyMet = True

        # check simple fixed bounds for controls.
        for control, bounds in self._config.safety.simpleControlBounds.items():
            if (controls[control] < bounds.lower) or (controls[control] > bounds.upper):
                safetyMet = False

        # check more complex, relational constraints.
        constr_sum = 0
        for control, value in controls.items():
            constr_sum = constr_sum + value
        if (constr_sum < self._config.safety.complexConstraints.addMin) or \
                (constr_sum > self._config.safety.complexConstraints.addMax):
            safetyMet = False

        return safetyMet
