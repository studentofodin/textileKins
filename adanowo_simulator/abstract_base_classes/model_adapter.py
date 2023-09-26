from abc import ABC, abstractmethod
import numpy as np


class AbstractModelAdapter(ABC):
    """Abstract class for a model adapter.

    A model adapter represents a model. Assume variable y depends on a variable x.
    * We use the term calculation if, given x, a value for y is directly returned.
    * We use the term model if, given x, a probability distribution for y is returned.
    To get a value you have to sample from the modeled distribution.

    More specifically, a model returns a mean and a variance (covariance matrix) resulting from uncertainties.
    The uncertainties can be divided into observation noise and model uncertainty.
    """

    @abstractmethod
    def predict_f(self, X: dict[str, float]) -> tuple[np.array, np.array]:
        """Applies the underlying model with inputs 'X' incorporating only model uncertainties (no observation noise).

        Returns
        -------
        np.array
            Mean.
        np.array
            Variance (covariance matrix).
        """
        pass

    @abstractmethod
    def predict_y(self, X: dict[str, float], **kwargs) -> tuple[np.array, np.array]:
        """Applies the underlying model.

        Parameters
        -------
        X : dict[str, float]
            Inputs of the model.
        **kwargs
            observation_noise_only : bool
                If not given or if False, observation noise and model uncertainty are incorporated.
                If True, only observation noise is incorporated.

        Returns
        -------
        np.array
            Mean.
        np.array
            Variance (covariance matrix).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        pass
