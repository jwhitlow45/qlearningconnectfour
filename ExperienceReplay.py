import numpy as np
from collections import deque

class Experience:
    def __init__(self, state: np.array, action: int, reward: int, done: bool, next_state: np.array) -> None:
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
        
    def sample(self, batch_size: int) -> np.array:
        sample_indexes = np.random.choice(len(self.memories), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip([self.memories[i] for i in sample_indexes])
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)