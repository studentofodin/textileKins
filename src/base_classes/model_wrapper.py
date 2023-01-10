import pathlib as pl
import dill
import pickle

from src.abstract_base_class.model_wrapper import AbstractModelWrapper
from src.base_classes.model_interface import *

class ModelWrapper(AbstractModelWrapper):

    @staticmethod
    def load_model(model_props: dict) -> AbstractModelInterface:
        with open(pl.Path(model_props["model_path"]), "rb") as file:
            pickle_obj = pickle.load(file)
        model_class = model_props["model_class"]
        if model_class == "SVGP":
            mdl = AdapterSVGP(pickle_obj, model_props, rescale_y=True)
        elif model_class == "GPy_GPR":
            mdl = AdapterGPy(pickle_obj, model_props, rescale_y=True)
        else:
            raise (TypeError(f"The model class {model_class} is not yet supported"))
        return mdl

    @property
    def model(self) -> AbstractModelInterface:
        return self._machineModel

    @machineModel.setter
    def machineModel(self, machineModel):
        self._machineModel = machineModel

    # def __init__(self):
    #     self.machineModel ModelInterface({
    #                                             "inputs": ["input_d", "input_c"],
    #                                             "output": "target_b"
    #                                         })

    def mapActionsToInputs(self, action):
        return np.array([3,2,1])

    def callMachineModel(self, input):
        return self.machineModel.calcMeanAndStd(np.array([5,4,3]), latent=True)

    def interpretModelOutputs(self, mean: np.array, lowerConfidence: np.array, upperConfidence: np.array) -> np.array:
        return [1,2,3]

    def getOutput(self, action):
        mean, lowerConfidence, upperConfidence = self.callMachineModel(input)
        output = self.interpretModelOutputs(mean, lowerConfidence, upperConfidence)
        return output