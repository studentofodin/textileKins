from abc import ABC, abstractmethod
import numpy as np
from omegaconf import DictConfig


class AbstractModelManager(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    @abstractmethod
    def get_model_outputs(self, inputs: dict[str, float]) -> tuple[np.array, dict[str, float]]:
        """
        get sampled output value from inputs for each model.
        return output values as numpy array and as dictionary.
        """
        pass

    @abstractmethod
    def update_model_allocation(self, changed_outputs: list[str]) -> None:
        """
        update the models for the outputs listed in changed_outputs based on the entries in the own config.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        reset to initial values.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        shutdown all processes.
        """
        pass

    @abstractmethod
    def _call_models(self, inputs: dict[str, float], latent=False) -> (dict[str, np.array], dict[str, np.array]):
        """
        get and return mean and variance prediction of output from inputs for each model.
        latent=True includes noise, latent=False not.
        """
        pass

    @abstractmethod
    def _sample_output_distribution(self, mean_pred: dict[str, np.array], var_pred: dict[str, np.array]) \
            -> (np.array, dict[str, float]):
        """
        sample output distribution dependent on corresponding mean and variance prediction
        (mean_pred, var_pred) for each model.
        Return output samples as numpy array and as dictionary.
        """
        pass

    @abstractmethod
    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        """
        load model corresponding to model_name.
        allocate this model to the output output_name.
        """
        pass
