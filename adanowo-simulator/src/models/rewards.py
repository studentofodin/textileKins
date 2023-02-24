import numpy as np

from src.abstract_base_class.reward import AbstractReward


class ProcessReward(AbstractReward):
    def __init__(self, requirements, weights, config):
        self._requirements = requirements
        self._weights = weights
        self._config = config

    def requirements(self) -> dict:
        return self._requirements

    def rewardValue(self) -> float:
        pass

    def calculateReward(self, currentState: dict, currentModelOutput: np.array,  safetyFlag: bool) -> float:
        target_a = currentState["target_a"]
        target_b = currentState["target_b"]
        disturbance_d = currentState["disturbance_d"]
        weight_b = self._weights["weight_b"]
        fibre_costs = self._config["fibre_costs"]
        reward = target_a*fibre_costs + weight_b*target_b - disturbance_d
        if self._requirements["b_lower"] < target_b < self._requirements["b_upper"] or safetyFlag:
            reward = - self._config["penalty"]
        return reward

    def calculatePenalty(self, state: np.array, modelOutput: np.array) -> float:
        return 0.0
