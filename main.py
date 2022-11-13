from ExperienceReplay import Experience, Memory
import tensorflow as tf
from keras import models
import numpy as np
from kaggle_environments import make

import Networks
from Agent import Agent
from board import BOARD_HEIGHT, BOARD_WIDTH, is_valid_move

# networks
target_net: tf.keras.Sequential = Networks.create_conv2d_model()
primary_net: tf.keras.Sequential = Networks.create_conv2d_model()

def main():
    global primary_net, target_net
    
    print('=======================================================================')
    print(tf.config.list_physical_devices('GPU'))
    print('=======================================================================')
    
    # SAVE PARAMETERS
    SAVE_FREQ = 1000
    RENDER_FREQ = 500
    RENDER_ENV = make('connectx', debug=True)
            
    # HYPERPARAMETERS
    GAMMA = 0.99
    BATCH_SIZE = 32    
    REPLAY_SIZE = 10000
    LEARNING_RATE = 1E-4
    OPTIMIZER = tf.optimizers.SGD(LEARNING_RATE)          
    SYNC_TARGET_FRAMES = 1000
    REPLAY_START_SIZE = 1000
    EPS_DECAY = .999985
    EPS_MIN = 0.02

    # persistent parameters
    memory = Memory(REPLAY_SIZE)
    agent = Agent(memory)
    total_rewards = []
    epsilon = 1
    frames = 0
    best_mean_reward = None
    
    output = ''
    
    while True:
        frames += 1
        epsilon = max(epsilon*EPS_DECAY, EPS_MIN)
        reward = agent.step_forward(primary_net, epsilon)
        
        if frames % SAVE_FREQ == 0:
            with open(f'./stats/stats-{frames}.csv', 'w') as FILE:
                FILE.write(output)
                output = ''
        
        if frames % RENDER_FREQ == 0:
            # render game from model
            RENDER_ENV.reset()
            RENDER_ENV.run([functional_agent, 'negamax'])
            game_render = RENDER_ENV.render(mode='html')
            with open(f'./models/render-{len(total_rewards)}-{frames}.html', 'w') as FILE:
                FILE.write(game_render)
        
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print(f'{frames}: {len(total_rewards)} games, mean reward {mean_reward}, epsilon {epsilon}')
            output += f'{frames},{len(total_rewards)},{mean_reward},{epsilon}\n'
        
            # save model when best reward is improved
            if best_mean_reward is None or best_mean_reward < mean_reward:
                models.save_model(primary_net, f'./models/dqn-{len(total_rewards)}-{frames}-best.h5')
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print(f"Best mean reward updated: {best_mean_reward}")
                
                # render game from model
                RENDER_ENV.reset()
                RENDER_ENV.run([functional_agent, 'negamax'])
                game_render = RENDER_ENV.render(mode='html')
                with open(f'./models/render-{len(total_rewards)}.html', 'w') as FILE:
                    FILE.write(game_render)
                    
                
        # # wait for memory to fill up before learning
        if len(memory) < REPLAY_START_SIZE:
                    continue
        
        with tf.GradientTape() as tape:
            # get batch of data from memory (experience replay)
            batch = agent.memory.sample(BATCH_SIZE)
            states, actions, rewards, dones, next_states = batch
            
            # convert action, reward, and done_mask lists to torch tensors
            actions_v = tf.convert_to_tensor(actions)
            rewards_v = tf.convert_to_tensor(rewards, dtype='float32')
            done_mask = tf.convert_to_tensor(dones)
            
            # reshape states to fit into model input
            states = np.array(states).reshape((-1, BOARD_HEIGHT, BOARD_WIDTH))
            next_states = np.array(next_states).reshape((-1, BOARD_HEIGHT, BOARD_WIDTH))
            
            # predict states using q and q'
            preds = primary_net(states)
            next_preds = target_net(next_states)
            
            # get q values for given actions from q prediction
            state_action_values = tf.gather(preds, actions_v, batch_dims=1, axis=1)
            # get max value of q' prediction
            next_state_values = tf.reduce_max(next_preds, axis=1)
            # apply done mask to set states which do not have a next state to 0 reward (end of game has no next state)
            next_state_values = tf.where(done_mask, next_state_values, tf.zeros((1)))
            # detach to prevent q' from being optimized by sgd
            next_state_values = tf.stop_gradient(next_state_values)
            
            # bellman approximation
            expected_state_action_values = tf.add_n(tf.scalar_mul(GAMMA, next_state_values), rewards_v)
            
            # calculate loss using bellman approximation
            loss = tf.losses.MSE(expected_state_action_values, state_action_values)

            # use loss to adjust gradients
            gradients = tape.gradient(loss, primary_net.trainable_variables)
            OPTIMIZER.apply_gradients(zip(gradients, primary_net.trainable_variables))
                
            if frames % SYNC_TARGET_FRAMES == 0:
                target_net.set_weights(primary_net.get_weights())

# functional agent for rendering games without epsilon decay
def functional_agent(observation, config):
    global primary_net, target_net
    
    flat_board = observation['board']
    # shape board for model input
    board = np.array(flat_board).reshape((1, BOARD_HEIGHT, BOARD_WIDTH))
    # make move based on board
    preds = primary_net(board)
    pred_list = list(preds[0].numpy())
    # create list of weighted moves and sort based on weight
    weighted_actions = [(weight, i) for i, weight in enumerate(pred_list)]
    weighted_actions.sort(key=lambda x: x[0], reverse=True)
    # find first valid move
    for _, action in weighted_actions:
        if is_valid_move(flat_board, action):
            return action
    return -1
    
if __name__ == '__main__':
    main()