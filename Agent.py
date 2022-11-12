import numpy as np
from keras import Sequential
from kaggle_environments import Environment
from ExperienceReplay import Memory, Experience
from board import isValidMove, BOARD_HEIGHT, BOARD_WIDTH, EMPTY

class Agent: 
    def __init__(self, env, memory: Memory):
        self.env = env
        self.memory = memory
        self._reset()
        
    def _reset(self):
            self.state = self.env.reset()
            self.total_reward = 0
            
    def count_empties(self):
        empties = 0
        board = self.state['board']
        for i in range(len(board)):
            if board[i] == EMPTY:
                empties += 1
        return empties
            
    def get_action(self, network: Sequential, epsilon: float):
        # get action
        if np.random.random() < epsilon: # take epsilon move
            action = np.random.choice(BOARD_WIDTH)
            while not isValidMove(self.state['board'], action):
                return np.random.choice(range(BOARD_WIDTH))
            
        # take "intelligent" move
        # make predictions based on baord state
        board = np.array(self.state['board'])
        board = board.reshape(1, BOARD_HEIGHT, BOARD_WIDTH)
        weights = list(network.predict(board, verbose=0)[0])
        
        # turn into weighted actinos and sort based on weight
        weighted_actions = [(weights[i], i) for i in range(len(weights))]
        weighted_actions.sort(key=lambda x: x[0], reverse=True)
        
        # iterate over weighted actions to find best valid move
        for _, action in weighted_actions:
            if isValidMove(self.state['board'], action):
                return action
        return -1
            
    def step_forward(self, network: Sequential, epsilon: float):
        action = self.get_action(network, epsilon)
        reward = None
        final_reward = None
        epsilon = 0
            
        # get resulting state and reward from an action
        new_state, reward_multiplier, is_done, _ = self.env.step(action)
        reward = self.count_empties() * reward_multiplier
        
        if reward_multiplier is not None:
            self.total_reward += reward
        
        # store in agent memory
        experience = Experience(self.state, action, reward, is_done, new_state)
        self.memory.append_memory(experience)
        
        if is_done:
            final_reward = self.total_reward
            self._reset()
        return final_reward