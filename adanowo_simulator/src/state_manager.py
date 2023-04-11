import numpy as np
from omegaconf import DictConfig
from typing import Dict

from src.abstract_base_class.state_manager import AbstractStateManager


class StateManager(AbstractStateManager):

    def __init__(self, config: DictConfig, actionType: int):
        self._initialconfig = config.copy()
        self._n_controls = len(config.initialControls)
        self._n_disturbances = len(config.disturbances)
        self._actionType = actionType
        self.reset()

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

    def calculateControlsFromAction(self, action: np.array) -> bool:
        updatedControls = dict()

        if self._actionType == 0:
            for index, control in enumerate(self._currentControls.keys()):
                updatedControls[control] = self._currentControls[control] + action[index]

        else:
            for index, control in enumerate(self._currentControls.keys()):
                updatedControls[control] = action[index]

        safetyFlag = self._safetyMet()
        if safetyFlag:
            self._currentControls = updatedControls
        currentState = self._currentControls | dict(self._config.disturbances)

        return currentState, safetyFlag

    def reset(self) -> Dict[str, float]:
        self._config = self._initialconfig.copy()
        self._currentControls = dict(self._config.initialControls)
        if not self._safetyMet():
            raise AssertionError("The initial setting is unsafe. Aborting Experiment.")
        currentState = self._currentControls | dict(self._config.disturbances)
        return currentState

    def _safetyMet(self) -> bool:

        safetyFlag = True

        # check simple fixed bounds for controls.
        for control, bounds in self._config.safety.simpleControlBounds.items():
            if (self._currentControls[control] < bounds.lower) or (self._currentControls[control] > bounds.upper):
                safetyFlag = False

        # check more complex, relational constraints.
        constr_sum = 0
        for control, value in self._currentControls.items():
            constr_sum = constr_sum + value
        if (constr_sum < self._config.safety.complexConstraints.addMin) or \
                (constr_sum > self._config.safety.complexConstraints.addMax):
            safetyFlag = False

        return safetyFlag
