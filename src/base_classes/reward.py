import numpy as np

from src.abstract_base_class.reward import AbstractReward


class Reward(AbstractReward):
    def __init__(self, config: "DictConfig"):
        self._config = dict(config)
        self._requirements = dict(config.requirements)
        self._weights = dict(config.weights)
        self._rewardValue = 0.0

    @property
    def config(self) -> dict:
        return self._config

    @property
    def requirements(self) -> dict:
        return self._requirements

    @property
    def weights(self) -> dict:
        return self._weights

    @property
    def rewardValue(self) -> float:
        return self._rewardValue

    def calculateReward(self, state: dict, observation: dict, safety_flag: bool) -> float:
        target_a = state["target_a"]
        target_b = state["target_b"]
        disturbance_d = state["disturbance_d"]
        weightB = self._weights["weightB"]
        fibreCosts = self._config["fibreCosts"]
        self._rewardValue = target_a*fibreCosts + weightB*target_b - disturbance_d
        if self._requirements["bLower"] < target_b < self._requirements["bUpper"] or safety_flag:
            self._rewardValue = - self._config["penalty"]
        return self._rewardValue