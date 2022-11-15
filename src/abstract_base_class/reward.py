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

    @abstractmethod
    def calculate_reward(self, current_state:np.array, current_model_output:np.array, safety_flag:bool) -> float :
        pass

    @abstractmethod
    def calculate_penalty(self) -> float :
        pass
