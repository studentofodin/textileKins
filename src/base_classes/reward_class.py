
import numpy as np
from abstract_base_class.reward import AbstractReward


class Reward(AbstractReward):
    def __init__(self, ITARequirements, rewardValue=0.0):
        self._ITARequirements = ITARequirements
        self._rewardValue = rewardValue

    @property
    def ITARequirements(self) -> dict :
        return self._ITARequirements

    @property
    def rewardValue(self) -> float :
        return self._rewardValue

    def calculateReward(self, currentState: np.array, currentModelOutput: np.array, safetyFlag: bool) -> float:
        print("Calculating reward")
        return 5.0

    def calculatePenalty(self, state: np.array, modelOutput: np.array) -> float:
        print("Penalty")
        return 2.0
