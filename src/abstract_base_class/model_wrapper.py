from abc import ABC, abstractmethod

import numpy as np

class ModelWrapper(ABC):

    @property
    @abstractmethod
    def wrapper(self) -> any:
        pass

    @abstractmethod
    def map_actions_to_inputs(self, action:np.array) -> np.array:
        pass

    @abstractmethod
    def interpret_model_outputs(self, mean:np.array, lower_confidence:np.array, upper_confidence:np.array) -> np.array:
        pass

    @abstractmethod
    def call_machine_model(self, input:np.array) -> [np.array, np.array, np.array]:
        pass
