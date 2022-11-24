import numpy as np
from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.base_classes.model_interface_class import ModelInterface


class ModelWrapperClass(AbstractModelWrapper):

    @property
    def machineModel(self) -> ModelInterface:
        return self._machineModel

    def __init__(self, action, machineModel):
        self._action = action
        self._machineModel = machineModel

    def mapActionsToInputs(self, action):
        return np.array([3,2,1])

    def callMachineModel(self, input):
        self.machineModel.calc_mean_and_std(np.array([5,4,3]), latent=True)
        return np.zeros(3), np.array([4,5,6]),np.array([6,5,4])

    def interpretModelOutputs(self, mean: np.array, lowerConfidence: np.array, upperConfidence: np.array) -> np.array:
        return [1,2,3]

    def getOutput(self, action):
        input = self.mapActionsToInputs(action)
        mean,lowerConfidence,upperConfidence = self.callMachineModel(input)
        output = self.interpretModelOutputs(mean,lowerConfidence,upperConfidence)
        return output
