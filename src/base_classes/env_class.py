
import numpy as np
from src.abstract_base_class.environment import AbstractITAEnvironment
from reward_class import Reward

class ITAEnvironment(AbstractITAEnvironment):

    def __init__(self, machine=True, max_steps=200, current_state=np.ones(3)):
        self._machine = machine
        self._reward = Reward({'key':2,'key2':3})
        self._tracker = 
        self._max_steps = max_steps
        self._current_state = current_state



    
