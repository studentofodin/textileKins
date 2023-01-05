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
        self.machineModel = ModelInterface({
                                                "inputs": ["input_d", "input_c"],
                                                "output": "target_b"
                                            })

    def mapActionsToInputs(self, action):
        return np.array([3,2,1])

    def callMachineModel(self, input):
        return self.machineModel.calcMeanAndStd(np.array([5,4,3]), latent=True)

    def interpretModelOutputs(self, mean: np.array, std) -> np.array:
        return np.random.normal(mean, std, 1000)

    def getOutput(self, action):
        input = self.mapActionsToInputs(action)
        mean, std = self.callMachineModel(input)
        output = self.interpretModelOutputs(mean, std)
        return output
