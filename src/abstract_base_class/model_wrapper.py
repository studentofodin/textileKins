from abc import ABC, abstractmethod

import numpy as np

from base_classes.model_interface_class import ModelInterface

class AbstractModelWrapper(ABC):

    @property
    @abstractmethod
    def machineModel(self) -> ModelInterface:
        pass

    @abstractmethod
    def mapActionsToInputs(self, action:np.array) -> np.array:
        pass

    @abstractmethod
    def interpretModelOutputs(self, mean:np.array, lowerConfidence:np.array, upperConfidence:np.array) -> np.array:
        pass

    @abstractmethod
    def callMachineModel(self, input:np.array) -> list[np.array, np.array, np.array]:
        pass

    @abstractmethod
    def getOutput(self, action:np.array) -> np.array:
        pass