from types import ModuleType, MethodType
import numpy as np

from adanowo_simulator.abstract_base_classes.calculation_adapter import AbstractCalculationAdapter


class CalculationAdapter(AbstractCalculationAdapter):

    def __init__(self, calculation_module: ModuleType) -> None:
        self._calculate: MethodType = calculation_module.calculate

    def calculate(self, X: dict[str, float]) -> np.array:
        c = np.array(self._calculate(X)).flatten()
        return c
