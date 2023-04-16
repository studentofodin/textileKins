from omegaconf import DictConfig

from src.abstract_base_class.reward_manager import AbstractRewardManager


class RewardManager(AbstractRewardManager):
    def __init__(self, config: DictConfig):
        self._initialConfig = config.copy()
        self.reset()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def reward_range(self) -> tuple[float, float]:
        return -float("inf"), float("inf")

    def getReward(self, state: dict[str, float], outputs: dict[str, float],
                                   safetyMet: bool) -> tuple[float, bool]:
        reqsMet = self._reqsMet(outputs)

        # penalty.
        if not (reqsMet and safetyMet):
            reward = -self._config.penalty

        # no penalty.
        else:
            targetA = outputs["min_area_weight"]
            targetB = outputs["unevenness_card_web"]
            cost = state["v_Arbeiter_HT"]
            w1 = self._config.weights.w1
            w2 = self._config.weights.w2
            fibreCosts = self._config.fibreSettings.fibreCosts
            reward = w1*fibreCosts*targetA + w2*targetB - cost

        return reward, reqsMet

    def reset(self) -> None:
        self._config = self._initialConfig.copy()

    def _reqsMet(self, outputs: dict[str, float]) -> bool:
        # assume that requirement constraints are met.
        reqsMet = True

        # check simple fixed bounds for outputs.
        for output, lowerBound in self._config.requirements.simpleOutputBounds.lower.items():
            if outputs[output] < lowerBound:
                reqsMet = False
        for output, upperBound in self._config.requirements.simpleOutputBounds.upper.items():
            if outputs[output] > upperBound:
                reqsMet = False

        # check more complex, relational constraints.
        product = 1
        for output, value in outputs.items():
            product = product * value
        if (product < self._config.requirements.complexConstraints.multMin) or \
                (product > self._config.requirements.complexConstraints.multMax):
            reqsMet = False

        return reqsMet


