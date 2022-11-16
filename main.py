from ExperienceReplay import Experience, Memory
import tensorflow as tf
from keras import models
import numpy as np
from kaggle_environments import make
import gc

import Networks
from Agent import Agent
from board import BOARD_HEIGHT, BOARD_WIDTH, is_valid_move

# networks
primary_net: tf.keras.Sequential = Networks.create_conv2d_model()
target_net: tf.keras.Sequential = Networks.create_conv2d_model()
target_net.set_weights(primary_net.get_weights())

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
    LEARNING_RATE = .0001
    OPTIMIZER = tf.optimizers.SGD(LEARNING_RATE)          
    SYNC_TARGET_FRAMES = 1000
    REPLAY_START_SIZE = 1000
    EPS_DECAY = .999985
    EPS_MIN = 0.02
    
    # TOGGLES
    ENABLE_RENDER = False
    ENABLE_SAVE = False

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
        
        if frames % SAVE_FREQ == 0 and ENABLE_SAVE:
            with open(f'./stats/stats-{frames}.csv', 'w') as FILE:
                FILE.write(output)
                output = ''
            models.save_model(primary_net, f'./models/dqn-{len(total_rewards)}-{frames}.h5')
            
        
        if frames % RENDER_FREQ == 0 and ENABLE_RENDER:
            # render game from model
            for cur_agent in ['random', 'negamax']:
                RENDER_ENV.reset()
                RENDER_ENV.run([functional_agent, cur_agent])
                game_render = RENDER_ENV.render(mode='html')
                with open(f'./models/render-{cur_agent}-{len(total_rewards)}-{frames}.html', 'w') as FILE:
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
                for cur_agent in ['random', 'negamax']:
                    RENDER_ENV.reset()
                    RENDER_ENV.run([functional_agent, cur_agent])
                    game_render = RENDER_ENV.render(mode='html')
                    with open(f'./models/render-{cur_agent}-{len(total_rewards)}-best.html', 'w') as FILE:
                        FILE.write(game_render)
                    
                
        # # wait for memory to fill up before learning
        if len(memory) < REPLAY_START_SIZE:
                    continue
        
        # get batch of data from memory (experience replay)
        batch = agent.memory.sample(BATCH_SIZE)
        states, actions, rewards, dones, next_states = batch

        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            with tf.GradientTape() as tape:
                if not done:
                    reward = 0
                # get best predicted action from target network
                next_state = np.array(next_state).reshape((-1, BOARD_HEIGHT, BOARD_WIDTH))
                target_pred = target_net(next_state).numpy().tolist()[0]
                action_prime = target_pred.index(max(target_pred))
                
                # calculate q*
                primary_future_pred = tf.gather(primary_net(next_state), tf.convert_to_tensor(action_prime), axis=1)
                q_star = tf.math.add(tf.scalar_mul(GAMMA, primary_future_pred), reward)

                # calulate q(s,a)
                state = np.array(state).reshape((-1, BOARD_HEIGHT, BOARD_WIDTH))
                q_theta = tf.gather(primary_net(state), tf.convert_to_tensor(action), axis=1)

                # calculate loss                
                loss = tf.square(tf.subtract(q_star, q_theta))
                # use loss to adjust gradients
                gradients = tape.gradient(loss, primary_net.trainable_variables)
                OPTIMIZER.apply_gradients(zip(gradients, primary_net.trainable_variables))
            
        garbage_collection()
        
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

def garbage_collection():
    gc.collect()
    tf.keras.backend.clear_session()
    
if __name__ == '__main__':
    main()