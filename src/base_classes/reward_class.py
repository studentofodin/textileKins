
import numpy as np
from src.abstract_base_class.reward import AbstractReward


class Reward(AbstractReward):
    def __init__(self, ITA_requirements, reward_value=0.0):
        self._ITA_requirements = ITA_requirements
        self._reward_value = reward_value

    @property
    def ITA_requirements(self) -> dict :
        return self._ITA_requirements

    @property
    def reward_value(self) -> float :
        return self._reward_value

    def calculate_reward(self, current_state: np.array, current_model_output: np.array, safety_flag: bool) -> float:
        print("Calculating reward")
        return 5.0

    def calculate_penalty(self, state: np.array, model_output: np.array) -> float:
        print("Penalty")
        return 2.0
