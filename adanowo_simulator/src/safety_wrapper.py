from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper
from omegaconf import DictConfig


class SafetyWrapper(AbstractSafetyWrapper):

    def __init__(self, config: DictConfig):
        self._initialconfig = config.copy()
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def reset(self) -> None:
        self._config = self._initialconfig.copy()

    def safetyMet(self, controls: dict[str, float]) -> bool:

        safetyFlag = True

        # check simple fixed bounds for controls.
        for control, bounds in self._config.safety.simpleControlBounds.items():
            if (controls[control] < bounds.lower) or (controls[control] > bounds.upper):
                safetyFlag = False

        # check more complex, relational constraints.
        constr_sum = 0
        for control, value in controls.items():
            constr_sum = constr_sum + value
        if (constr_sum < self._config.safety.complexConstraints.addMin) or \
                (constr_sum > self._config.safety.complexConstraints.addMax):
            safetyFlag = False

        return safetyFlag
