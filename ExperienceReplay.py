import numpy as np
from collections import deque
from typing import Tuple

class Experience:
    def __init__(self, state, action: int, reward: int, done: bool, next_state: np.array) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state
        
class Memory:
    def __init__(self, memory_size: int) -> None:
        self.memories = deque(maxlen=memory_size)
        
    def __len__(self):
        return len(self.memories)
        
    def append_memory(self, experience: Experience) -> None:
        self.memories.append(experience)
        
    def sample(self, batch_size: int) -> Tuple[np.array]:
        indexes = np.random.choice(len(self.memories), batch_size, replace=False)
        states = [self.memories[i].state['board'] for i in indexes]
        actions = [self.memories[i].action for i in indexes]
        rewards = [self.memories[i].reward for i in indexes]
        dones = [self.memories[i].done for i in indexes]
        next_states = [self.memories[i].next_state['board'] for i in indexes]
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)