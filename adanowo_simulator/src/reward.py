from omegaconf import DictConfig

from src.abstract_base_class.reward import AbstractReward


class Reward(AbstractReward):
    def __init__(self, config: DictConfig):
        self._initialconfig = config.copy()
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def reward_range(self) -> (float, float):
        return -float("inf"), float("inf")

    def calculateRewardAndReqsFlag(self, state: dict[str, float], outputs: dict[str, float],
                                   safetyFlag: bool) -> tuple[float, bool]:

        reqsFlag = self._reqsMet(outputs)

        if not (reqsFlag and safetyFlag):  # penalty
            rewardValue = -self._config.penalty

        else:  # no penalty
            targetA = outputs["min_area_weight"]
            targetB = outputs["unevenness_card_web"]
            cost = state["v_Arbeiter_HT"]
            w1 = self._config.weights.w1
            w2 = self._config.weights.w2
            fibreCosts = self._config.fibreSettings.fibreCosts
            rewardValue = w1*fibreCosts*targetA + w2*targetB - cost

        self._rewardValue = rewardValue

        return rewardValue, reqsFlag

    def reset(self) -> None:
        self._config = self._initialconfig.copy()

    def _reqsMet(self, outputs: dict[str, float]) -> bool:
        reqsFlag = True

        # check simple fixed bounds for outputs.
        for output, lowerBound in self._config.requirements.simpleOutputBounds.lower.items():
            if outputs[output] < lowerBound:
                reqsFlag = False
        for output, upperBound in self._config.requirements.simpleOutputBounds.upper.items():
            if outputs[output] > upperBound:
                reqsFlag = False

        # check more complex, relational constraints.
        product = 1
        for output, value in outputs.items():
            product = product * value
        if (product < self._config.requirements.complexConstraints.multMin) or (product > self._config.requirements.complexConstraints.multMax):
            reqsFlag = False

        self._reqsFlag = reqsFlag

        return reqsFlag


