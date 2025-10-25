import numpy as np
import gymnasium as gym
import pandas as pd
import random
import sys
sys.path.insert(0, './even_more_complex_problem')
import even_more_complex_problem_env

row_nums={
    1:  {1,3},
    2:  {2,4,5,12,20,17},
    3:  {6,21},
    4:  {6,10},
    5:  {2,4,8,13,18},
    6:  6,
    7:  7,
    8:  {2,4,14},
    9:  9,
    10: 10,
    11: 11,
    12: {2,4},
    13: {2,4,8,13,18},
    14: 14,
    15: 15,
    16: 16,
    17: {8,13,18},
    19: 19,
    21: 21,
}

def get_action(q_table, is_opponent_lost, row_num, epsilon):
    if is_opponent_lost:
        return 1
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
        
        is_q_1_lost = False
        is_q_2_lost = False
        
        while not (terminated or truncated):
            if curr_symbol == "a":
                
                agent_id=1
                agent_1_row_num = 21 + agent_1_observation if not is_q_2_lost else agent_1_observation
                
                if q_1_update_delayed:
                    # Q-value update for agent 1
                    q_1[agent_1_prev_row_num][agent_communicate_1] += alpha * (reward_1 + gamma * np.max(q_1[agent_1_row_num]) - q_1[agent_1_prev_row_num][agent_communicate_1])
                    q_1_update_delayed = False
                    reward_1 = 0
                
                agent_communicate_1 = get_action(q_1, is_q_2_lost, agent_1_row_num, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_communicate_1))
                
                _, agent_1_observation, agent_2_observation = config
                
                is_q_2_lost = agent_2_observation == -1
                    
                reward_1 += reward
                
                curr_symbol=info['input_alphabet']
                
                q_1_update_delayed = True
                agent_1_prev_row_num = agent_1_row_num
                            
            if curr_symbol == "b":
                agent_id=2
                agent_2_row_num = 21 + agent_2_observation if not is_q_1_lost else agent_2_observation

                
                if q_2_update_delayed:
                    # Q-value update for agent 2
                    q_2[agent_2_prev_row_num][agent_communicate_2] += alpha * (reward_2 + gamma * np.max(q_2[agent_2_row_num]) - q_2[agent_2_prev_row_num][agent_communicate_2])
                    q_2_update_delayed = False
                    reward_2 = 0
                
                agent_communicate_2 = get_action(q_2, is_q_1_lost, agent_2_row_num, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_communicate_2))
                
                _, agent_1_observation, agent_2_observation = config
                
                is_q_1_lost = agent_1_observation == -1
                
                reward_2 += reward
                
                curr_symbol=info['input_alphabet']
                
                q_2_update_delayed = True
                agent_2_prev_row_num = agent_2_row_num
        
        reward_1 += reward
        reward_2 += reward
        
        # Final Q-value updates
        q_1[agent_1_prev_row_num][agent_communicate_1] += alpha * (reward_1 + gamma * 0 - q_1[agent_1_prev_row_num][agent_communicate_1])
        q_2[agent_2_prev_row_num][agent_communicate_2] += alpha * (reward_2 + gamma * 0 - q_2[agent_2_prev_row_num][agent_communicate_2])

q_training_env = gym.make('EvenMoreComplexEnv-v0', render_mode=None)

q_1 = np.zeros(43, q_training_env.action_space.n)
q_2 = np.zeros(43, q_training_env.action_space.n)

q_training(q_training_env, q_1, q_2, epochs=1000000, alpha=0.01, gamma=0.5, epsilon=0.1)

q_1_df = pd.DataFrame(q_1, columns=["communicate", "do not communcate"])
q_2_df = pd.DataFrame(q_2, columns=["communicate", "do not communcate"])

q_1_df.to_csv(f'even_more_complex_problem/demo_q1_table.csv')
q_2_df.to_csv(f'even_more_complex_problem/demo_q2_table.csv')

# Training done, go to simulation.py for simulation