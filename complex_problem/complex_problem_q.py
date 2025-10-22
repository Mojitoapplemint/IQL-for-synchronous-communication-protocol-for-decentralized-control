import numpy as np
import gymnasium as gym
import pandas as pd
import random
import sys
sys.path.insert(0, './complex_problem')
import complex_problem_env

agent_1_row_nums = {
    (0, 'a'):0,
    (1, 'a'):1,
    (2, 'a'):2,
    (3,'a'):3,
    (4, 'a'):4,
    (5, 'a'):5,
    (-1, 'a'):6,
}

agent_2_row_nums = {
    (0, 'b'):0,
    (1, 'b'):1,
    (2, 'b'):2,
    (3,'b'):3,
    (4, 'b'):4,
    (5, 'b'):5,
    (-1, 'b'):6,
}

def get_action(q_table, row_num, epsilon):
    if random.uniform(0, 1) < epsilon:
        return np.argmin(q_table[row_num]) # Explore: choose the action that is not best
    else:
        return  np.argmax(q_table[row_num])  # Exploit: best action from Q-table

def q_training(env, q_1, q_2, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1):

    for epoch in range(epochs):
        if (epoch%100==0):
            print(str(100*epoch/epochs)+"%","done" , end="\r")
        
        config, info = env.reset()
        
        curr_symbol=info['input_alphabet']
        
        _, agent_1_observation, agent_2_observation = config
        
        agent_1_prev_row_num = -1
        agent_2_prev_row_num = -1
        
        terminated = False
        truncated = False
        
        agent_communicate_1 = -1
        agent_communicate_2 = -1
        
        q_1_update_delayed = False
        q_2_update_delayed = False
        
        reward_1 = 0
        reward_2 = 0
        
        while not (terminated or truncated):
            if curr_symbol == "a":
                
                agent_id=1
                agent_1_row_num = agent_1_row_nums[(agent_1_observation, curr_symbol)]
                
                if q_1_update_delayed:
                    # Q-value update for agent 1
                    q_1[agent_1_prev_row_num][agent_communicate_1] += alpha * (reward_1 + gamma * np.max(q_1[agent_1_row_num]) - q_1[agent_1_prev_row_num][agent_communicate_1])
                    q_1_update_delayed = False
                    reward_1 = 0
                
                agent_communicate_1 = get_action(q_1, agent_1_row_num, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_communicate_1))
                
                _, agent_1_observation, agent_2_observation = config
                
                reward_1 += reward
                
                curr_symbol=info['input_alphabet']
                
                q_1_update_delayed = True
                agent_1_prev_row_num = agent_1_row_num
                            
            if curr_symbol == "b":
                agent_id=2
                agent_2_row_num = agent_2_row_nums[(agent_2_observation, curr_symbol)]
                
                if q_2_update_delayed:
                    # Q-value update for agent 2
                    q_2[agent_2_prev_row_num][agent_communicate_2] += alpha * (reward_2 + gamma * np.max(q_2[agent_2_row_num]) - q_2[agent_2_prev_row_num][agent_communicate_2])
                    q_2_update_delayed = False
                    reward_2 = 0
                
                agent_communicate_2 = get_action(q_2, agent_2_row_num, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_communicate_2))
                
                _, agent_1_observation, agent_2_observation = config
                
                reward_2 += reward
                
                curr_symbol=info['input_alphabet']
                
                q_2_update_delayed = True
                agent_2_prev_row_num = agent_2_row_num
        
        reward_1 += reward
        reward_2 += reward
        
        # Final Q-value updates
        q_1[agent_1_prev_row_num][agent_communicate_1] += alpha * (reward_1 + gamma * 0 - q_1[agent_1_prev_row_num][agent_communicate_1])
        q_2[agent_2_prev_row_num][agent_communicate_2] += alpha * (reward_2 + gamma * 0 - q_2[agent_2_prev_row_num][agent_communicate_2])

q_training_env = gym.make('ComplexEnv-v0', render_mode=None, string_mode="full")

q_1 = np.zeros((len(agent_1_row_nums), q_training_env.action_space.n))
q_2 = np.zeros((len(agent_2_row_nums), q_training_env.action_space.n))

q_training(q_training_env, q_1, q_2, epochs=1000000, alpha=0.01, gamma=0.1, epsilon=0.1)

q_1_df = pd.DataFrame(q_1, columns=["do not communcate", "communicate"])
q_2_df = pd.DataFrame(q_2, columns=["do not communcate", "communicate"])

q_1_df.to_csv(f'complex_problem/demo_q1_table.csv')
q_2_df.to_csv(f'complex_problem/demo_q2_table.csv')

# Training done, go to simulation.py for simulation