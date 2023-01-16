from src.abstract_base_class.safety_wrapper import AbstractSafetyWrapper


class SafetyWrapper(AbstractSafetyWrapper):

    def __init__(self, config: "DictConfig"):
        self._constraints = dict(config.constraints)

    @property
    def constraints(self) -> dict:
        return self._constraints

    def isWithinConstraints(self, state: dict) -> bool:
        return True




