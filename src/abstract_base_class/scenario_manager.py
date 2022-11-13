from abc import ABC, abstractmethod
import numpy as np

class AbstractScenarioManager(ABC):

    @property
    @abstractmethod
    def disturbance_setting(self) -> dict:
        pass

    @property
    @abstractmethod
    def fibre_setting(self) -> dict:
        pass

    @abstractmethod
    def set_disturbance(self, disturbance_setting):
        pass

    @abstractmethod
    def set_fibre_mixture(self, fibre_setting):
        pass