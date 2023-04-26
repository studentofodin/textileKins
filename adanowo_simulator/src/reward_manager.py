from omegaconf import DictConfig

from src.abstract_base_class.reward_manager import AbstractRewardManager


class RewardManager(AbstractRewardManager):
    def __init__(self, config: DictConfig):
        self._initial_config = config.copy()
        self.reset()

        self._config = None

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    @property
    def reward_range(self) -> tuple[float, float]:
        return -float("inf"), float("inf")

    def get_reward(self, state: dict[str, float], outputs: dict[str, float], safety_met: bool) -> tuple[float, bool]:
        reqs_met = self._reqs_met(outputs)

        # penalty.
        if not (reqs_met and safety_met):
            reward = -self._config.penalty

        # no penalty.
        else:
            target_a = outputs["min_area_weight"]
            target_b = outputs["unevenness_card_web"]
            cost = state["v_Arbeiter_HT"]
            w1 = self._config.weights.w1
            w2 = self._config.weights.w2
            fibre_costs = self._config.fibre_settings.fibre_costs
            reward = w1*fibre_costs*target_a + w2*target_b - cost

        return reward, reqs_met

    def reset(self) -> None:
        self._config = self._initial_config.copy()

    def _reqs_met(self, outputs: dict[str, float]) -> bool:
        # assume that requirement constraints are met.
        reqs_met = True

        # check simple fixed bounds for outputs.
        for output_name, lower_bound in self._config.requirements.simple_output_bounds.lower.items():
            if outputs[output_name] < lower_bound:
                reqs_met = False
        for output_name, upper_bound in self._config.requirements.simple_output_bounds.upper.items():
            if outputs[output_name] > upper_bound:
                reqs_met = False

        # check more complex, relational constraints.
        product = 1
        for output_name, value in outputs.items():
            product = product * value
        if (product < self._config.requirements.complex_constraints.mult_min) or \
                (product > self._config.requirements.complex_constraints.mult_max):
            reqs_met = False

        return reqs_met
