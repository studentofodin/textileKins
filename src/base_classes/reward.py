import numpy as np

from src.abstract_base_class.reward import AbstractReward


class Reward(AbstractReward):
    def __init__(self, requirements, rewardValue=0.0):
        self._requirements = requirements
        self._rewardValue = rewardValue

    @property
    def requirements(self) -> dict:
        return self._requirements

    @property
    def rewardValue(self) -> float:
        return self._rewardValue

    def calculateReward(self, currentState: np.array, currentModelOutput: np.array, safetyFlag: bool) -> float:
        print("Calculating reward")
        return 5.0

    def calculatePenalty(self, state: np.array, modelOutput: np.array) -> float:
        return 0.0
