from omegaconf import DictConfig
from typing import Tuple
from typing import Dict

from src.abstract_base_class.reward import AbstractReward


class Reward(AbstractReward):
    def __init__(self, config: DictConfig):
        self._config = config
        self._rewardValue = 0.0
        self._reqsFlag = True

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def rewardValue(self) -> float:
        return self._rewardValue

    @property
    def reqsFlag(self) -> bool:
        return self._reqsFlag

    def reqsMet(self, outputs: Dict[str, float]) -> bool:

        reqsFlag = True

        # check simple fixed bounds for outputs.
        for output, lowerBound in self._config.simpleOutputBounds.lower.items():
            if outputs[output] < lowerBound:
                reqsFlag = False
        for output, upperBound in self._config.simpleOutputBounds.upper.items():
            if outputs[output] > upperBound:
                reqsFlag = False

        # check more complex, relational constraints.
        product = 1
        for output, value in outputs.items():
            product = product * value
        if (product < self._config.complexConstraints.multMin) or (product > self._config.complexConstraints.multMax):
            reqsFlag = False

        self._reqsFlag = reqsFlag

        return reqsFlag

    def calculateRewardAndReqsFlag(self, state: Dict[str, float], outputs: Dict[str, float],
                                   safetyFlag: bool) -> Tuple[float, bool]:

        reqsFlag = self.reqsMet(outputs)

        # penalty.
        if not (reqsFlag and safetyFlag):
            rewardValue = -self._config.penalty

        # no penalty.
        else:
            targetA = outputs["min_area_weight"]
            targetB = outputs["unevenness_card_web"]
            cost = state["v_Arbeiter_HT"]
            w1 = self._config.weights.w1
            w2 = self._config.weights.w2
            fibreCosts = self._config.fibreSettings.fibreCosts
            rewardValue = w1*fibreCosts*targetA + w2*targetB - cost

        self._rewardValue = rewardValue

        return rewardValue, reqsFlag
