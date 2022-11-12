import numpy as np
from keras import Sequential
from kaggle_environments import Environment
from ExperienceReplay import Memory, Experience
from board import isValidMove, BOARD_HEIGHT, BOARD_WIDTH

class Agent: 
    def __init__(self, env, memory: Memory):
        self.env = env
        self.memory = memory
        self._reset()
        
    def _reset(self):
            self.state = self.env.reset()
            self.total_reward = 0
            
    def step_forward(self, network: Sequential, epsilon: float):
        action = None
        reward = None
        final_reward = None
        epsilon = 0

        # get action
        if np.random.random() < epsilon: # take epsilon move
            action = np.random.choice(BOARD_WIDTH)
            while not isValidMove(self.state, action):
                action = np.random.choice(range(BOARD_WIDTH))
        else: # take "intelligent" move
            board = np.array(self.state['board'])
            board = board.reshape(1, BOARD_HEIGHT, BOARD_WIDTH)
            preds = list(network.predict(board, verbose=0)[0])
            action = preds.index(max(preds))
            
        # get resulting state and reward from an action
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        
        # store in agent memory
        experience = Experience(self.state, action, reward, is_done, new_state)
        self.memory.append_memory(experience)
        
        if is_done:
            final_reward = self.total_reward
            self._reset()
        return final_reward