from abc import ABC, abstractmethod
import numpy as np

class AbstractReward(ABC):


    @property
    @abstractmethod
    def ITA_requirements(self) -> dict :
        pass

    @property
    @abstractmethod
    def reward_value(self) -> float :
        pass

    # @ITA_requirements.setter
    # @abstractmethod
    # def ITA_requirements(self, requirements:dict) :
    #     pass

    @reward_value.setter
    @abstractmethod
    def reward_value(self, reward_value:float):
        pass

    @abstractmethod
    def calculate_reward(self, current_state:np.array, current_model_output:np.array, safety_flag:bool) -> float :
        pass

    @abstractmethod
    def penalty(self) -> float :
        pass


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

    @ITA_requirements.setter
    def ITA_requirements(self, requirements:dict) :
        self._ITA_requirements = requirements

    @reward_value.setter
    def reward_value(self, reward_value:float):
        self._reward_value = reward_value

    def calculate_reward(self, current_state: np.array, current_model_output: np.array, safety_flag: bool) -> float:
        print("Calculating reward")
        return 5.0

    def penalty(self) -> float:
        print("penaltyyyy")
        return 2.0

def main():
    requirements = {'a': 1, 'b': 2}
    reward = Reward(requirements)
    reward.reward_value=5
    print(reward.ITA_requirements)
    print(reward.penalty())
    print(reward.reward_value)
    print(reward._reward_value)

main()
