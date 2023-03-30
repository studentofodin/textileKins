from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from omegaconf import DictConfig


class SafetyWrapper(AbstractSafetyWrapper):

    def __init__(self, config: DictConfig):
        self._config = config
        self._safetyFlag = True

    @property
    def safetyFlag(self) -> bool:
        return self._safetyFlag

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def safetyMet(self, controls: dict[str, float]) -> bool:

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
