from kaggle_environments import Environment
from ExperienceReplay import Memory

class Agent: 
    def __init__(self, env: Environment, memory: Memory):
        self.env = env
        self.memory = memory
        self._reset()
        
    def _reset(self):
            self.state = self.env.reset()
            self.total_reward = 0.0