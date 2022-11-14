from abc import ABC, abstractmethod
import numpy as np

class AbstractSafetyWrapper(ABC):

    @property
    @abstractmethod
    def constraints(self) -> dict:
        pass

    @abstractmethod
    def is_within_constraints(self) -> bool:
        pass

    @abstractmethod
    def calculate_clipped_state(self) :
        pass

    @abstractmethod
    def load_constraints(self, config_file):
        pass
