import numpy as np

from src.abstract_base_class.reward import AbstractReward


class Reward(AbstractReward):
    def __init__(self, config):
        self._config = config
        self._requirements = config.requirements
        self._weights = config.weights
        self._rewardValue = 0.0

    @property
    def requirements(self) -> dict:
        return self._requirements

    @property
    def rewardValue(self) -> float:
        return self._rewardValue

    def calculateReward(self, currentState: dict, currentModelOutput: np.array, safetyFlag: bool) -> float:
        target_a = currentState["input1"]
        target_b = currentState["input2"]
        disturbance_d = currentState["input3"]
        weightB = self._weights["output2"]
        fibreCosts = self._config.fibreCosts
        reward = target_a*fibreCosts + weightB*target_b - disturbance_d
        if self.requirements.bLower < target_b < self.requirements.bUpper or safetyFlag:
            reward = - self._config.penalty  
        return reward

    def calculatePenalty(self, state: dict, modelOutput: np.array) -> float:
        return 0.0