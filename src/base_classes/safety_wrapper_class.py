
import numpy as np
from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper


class SafetyWrapperClass(AbstractSafetyWrapper):

    @property
    def constraints(self) -> dict:
        return self._constraints

    def isWithinConstraints(self) -> bool:
        return True

    def calculateClippedState(self):
        return np.ones(3)

    def __init__(self, constraints):
        self._constraints=constraints


