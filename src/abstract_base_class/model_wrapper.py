from abc import ABC, abstractmethod

import numpy as np

class ModelWrapper(ABC):

    @property
    @abstractmethod
    def wrapper(self) -> any:
        pass

    @abstractmethod
    def mapActionsToInputs(self, action:np.array) -> np.array:
        pass

    @abstractmethod
    def interpretModelOutputs(self, mean:np.array, lowerConfidence:np.array, upperConfidence:np.array) -> np.array:
        pass

    @abstractmethod
    def callMachineModel(self, input:np.array) -> [np.array, np.array, np.array]:
        pass
