from abc import ABC, abstractmethod
import numpy as np

class AbstractConfiguration(ABC):

    @property
    @abstractmethod
    def requirements(self) -> dict:
        pass

    @property
    @abstractmethod
    def actor_constraints(self) -> dict:
        pass

    @property
    @abstractmethod
    def production_scenario(self) -> dict:
        pass

    @property
    @abstractmethod
    def action_params(self) -> dict:
        pass

    @property
    @abstractmethod
    def state_params(self) -> dict:
        pass

    @property
    @abstractmethod
    def steps_until_lab_data_available(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_params(self) -> dict:
        pass  