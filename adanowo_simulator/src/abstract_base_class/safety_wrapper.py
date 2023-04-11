from abc import ABC, abstractmethod


class AbstractSafetyWrapper(ABC):

    @abstractmethod
    def safetyMet(self, controls: dict[str, float]) -> bool:
        """
        check if the controls are within safety constraints.
        return this value and assign it to property safetyFlag.
        """
        pass

