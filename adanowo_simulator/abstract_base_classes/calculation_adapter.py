import numpy as np
from abc import ABC, abstractmethod


class AbstractCalculationAdapter(ABC):
    """Abstract class for a calculation adapter.

    A calculation adapter performs calculations. Assume variable y depends on a variable x.
    * We use the term calculation if, given x, a value for y is directly returned.
    * We use the term model if, given x, a probability distribution for y is returned.
    To get a value you have to sample from the modeled distribution."""

    @abstractmethod
    def calculate(self, X: dict[str, float]) -> np.array:
        """Performs a calculation."""
        pass
