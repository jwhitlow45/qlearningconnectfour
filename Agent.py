import numpy as np
from keras import Sequential
from kaggle_environments import make
from ExperienceReplay import Memory, Experience
from board import is_valid_move, count_empties, BOARD_HEIGHT, BOARD_WIDTH

class Agent: 
    def __init__(self, memory: Memory):
        self.maker = make('connectx', debug=True)
        self.env = self.maker.train([None, 'random'])
        self.memory = memory
        self._reset()
        
    def _reset(self):
            self.state = self.env.reset()
            self.total_reward = 0
            
    def get_action(self, network: Sequential, epsilon: float):
        # get action
        if np.random.random() < epsilon: # take epsilon move
            action = np.random.choice(BOARD_WIDTH)
            while not is_valid_move(self.state['board'], action):
                action = np.random.choice(BOARD_WIDTH)
            return action
            
        # take "intelligent" move
        # make predictions based on baord state
        board = np.array(self.state['board'])
        board = board.reshape(1, BOARD_HEIGHT, BOARD_WIDTH)
        weights = list(network.predict(board, verbose=0)[0])
        
        # turn into weighted actinos and sort based on weight
        weighted_actions = [(weights[i], i) for i in range(len(weights))]
        weighted_actions.sort(key=lambda x: x[0], reverse=True)
        print(weighted_actions)
        
        # iterate over weighted actions to find best valid move
        for _, action in weighted_actions:
            if is_valid_move(self.state['board'], action):
                return action
        return -1
            
    def step_forward(self, network: Sequential, epsilon: float):
        action = self.get_action(network, epsilon)
        reward = None
        final_reward = None
            
        # get resulting state and reward from an action
        new_state, reward_multiplier, is_done, _ = self.env.step(action)
        if reward_multiplier is not None:
            reward = count_empties(new_state['board']) * reward_multiplier
            self.total_reward += reward
        
        # store in agent memory
        experience = Experience(self.state, action, reward, is_done, new_state)
        self.memory.append_memory(experience)
        self.state = new_state
        
        if is_done:
            final_reward = self.total_reward
            # agents = None
            # if np.random.random() < epsilon:
            #     agents = [None, 'random']
            #     print('next opponent is random')
            # else:
            #     agents = [None, 'negamax']
            #     print('next opponent is negamax')
            # self.env = self.maker.train(agents)
            self._reset()
        return final_reward