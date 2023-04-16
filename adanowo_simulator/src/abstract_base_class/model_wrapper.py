from abc import ABC, abstractmethod
import numpy as np
from omegaconf import DictConfig

class AbstractModelWrapper(ABC):

    @property
    @abstractmethod
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    @abstractmethod
    def config(self, c):
        self._config = c

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        return self._n_outputs

    @abstractmethod
    def get_outputs(self, inputs: dict[str, float]) -> tuple[np.array, dict[str, float]]:
        """
        get sampled output value from inputs for each model.
        return output values as numpy array and as dictionary.
        """
        pass

    @abstractmethod
    def update(self, changed_outputs: list[str]) -> None:
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
    def _call_models(self, inputs: dict[str, float], latent=False) -> (dict[str, float], dict[str, float]):
        """
        get and return mean and standard deviation prediction of output from inputs for each model.
        latent=True includes noise, latent=False not.
        """
        pass

    @abstractmethod
    def _sample_output_distribution(self, mean_pred: dict[str, float], std_pred: dict[str, float]) \
            -> (np.array, dict[str, float]):
        """
        sample output distribution dependent on corresponding mean and standard deviation prediction (mean_pred, std_pred) for each model.
        return output samples as numpy array and as dictionary.
        """
        pass

    @abstractmethod
    def _allocate_model_to_output(self, output_name: str, model_name: str) -> None:
        """
        load model corresponding to model_name.
        allocate this model to the output output_name.
        """
        pass
