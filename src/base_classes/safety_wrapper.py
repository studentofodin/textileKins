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
        for control, bounds in self._config.simpleControlBounds.items():
            if (controls[control] < bounds.lower) or (controls[control] > bounds.upper):
                safetyFlag = False

        # check more complex, relational constraints.
        sum = 0
        for control, value in controls.items():
            sum = sum + value
        if (sum < self._config.complexConstraints.addMin) or (sum > self._config.complexConstraints.addMax):
            safetyFlag = False

        self._safetyFlag = safetyFlag

        return safetyFlag




