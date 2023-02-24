from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from omegaconf import DictConfig
from typing import Dict


class SafetyWrapper(AbstractSafetyWrapper):

    def __init__(self, config: DictConfig):
        self._config = config

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def safetyFlag(self) -> bool:
        return self._safetyFlag

    def safetyMet(self, controls: Dict[str, float]) -> bool:

        safetyFlag = True

        # check simple fixed bounds for controls.
        for control, bounds in self._config.safetyBounds.simpleControlBounds.items():
            if (controls[control] < bounds.lower) or (controls[control] > bounds.upper):
                safetyFlag = False

        # check more complex, relational constraints.
        constr_sum = 0
        for control, value in controls.items():
            constr_sum = constr_sum + value
        if (constr_sum < self._config.safetyBounds.complexConstraints.addMin) or \
                (constr_sum > self._config.safetyBounds.complexConstraints.addMax):
            safetyFlag = False

        self._safetyFlag = safetyFlag

        return safetyFlag
