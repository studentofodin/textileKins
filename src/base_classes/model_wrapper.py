import numpy as np
from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.base_classes.model_interface import ModelInterface


class ModelWrapper(AbstractModelWrapper):

    @property
    def machineModel(self) -> ModelInterface:
        return self._machineModel

    @machineModel.setter
    def machineModel(self, machineModel):
        self._machineModel = machineModel

    def __init__(self):
        self.machineModel = ModelInterface({'abc':7})

    def mapActionsToInputs(self, action):
        return np.array([3,2,1])

    def callMachineModel(self, input):
        self.machineModel.calcMeanAndStd(np.array([5,4,3]), latent=True)
        return np.zeros(3), np.array([4,5,6]),np.array([6,5,4])

    def interpretModelOutputs(self, mean: np.array, lowerConfidence: np.array, upperConfidence: np.array) -> np.array:
        return [1,2,3]

    def getOutput(self, action):
        input = self.mapActionsToInputs(action)
        mean, lowerConfidence, upperConfidence = self.callMachineModel(input)
        output = self.interpretModelOutputs(mean, lowerConfidence, upperConfidence)
        return output
