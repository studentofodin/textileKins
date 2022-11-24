
from gym import Env, spaces
import numpy as np
from src.abstract_base_class.environment import AbstractITAEnvironment
from reward_class import Reward
from src.abstract_base_class.experiment_tracking import ExperimentTracking
from src.abstract_base_class.interfaces import ModelInterface
from src.base_classes.model_wrapper_class import ModelWrapperClass

class ITAEnvironment(AbstractITAEnvironment, Env):

    def __init__(self, machine, max_steps=200, current_state=np.ones(3)):

        self._machine = ModelWrapperClass({'key3':7,'key4':8})
        self._reward = Reward({'key1':2,'key2':3})
        self._experiment_tracker = ExperimentTracking({'key5':3, 'key6':6})
        self._max_steps = max_steps
        self._current_state = current_state
        self.action_space = spaces.Box(self.low, self.high, np.dtype)
        self.observation_space
        self.spec
        self.metadata
        self.np_random 
    
    def step(self, action: ActionType) -> Tuple[_OperationType, float, bool, Dict[str, Any]]:
        return super().step(action)

    def reset(self) -> Any:
        return super().reset()
    
    def render():
        pass

    def close(self) -> None:
        return super().close()




    
