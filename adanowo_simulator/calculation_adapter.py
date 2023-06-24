import numpy as np
from abc import ABC, abstractmethod
from types import ModuleType

from adanowo_simulator.abstract_base_classes.calculation_adapter import AbstractCalculationAdapter

class CalculationAdapter(AbstractCalculationAdapter):

    def __init__(self, calculation_module: ModuleType) -> None:
        self._calculate = calculation_module.calculate

    def calculate(self, X: dict[str, float]) -> np.array:
        c = self._calculate(X)
        return c