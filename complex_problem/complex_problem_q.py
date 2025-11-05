import numpy as np
import gymnasium as gym
import pandas as pd
import random
import sys
sys.path.insert(0, './complex_problem')
import complex_problem_env

row_nums = {
    (False, 0 ):0,
    (False, 1 ):1,
    (False, 2 ):2,
    (False, 3 ):3,
    (False, 4 ):4,
    (False, 5 ):5,
    (False,-1 ):6,
    (True,  0 ):7,
    (True,  1 ):8,
    (True,  2 ):9,
    (True,  3 ):10,
    (True,  4 ):11,
    (True,  5 ):12,
    (True, -1 ):13,
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
        
        agent_1_communicate = -1
        agent_2_communicate = -1
        
        reward_1 = 0
        reward_2 = 0
        
        agent_1_in_dead_state = False
        agent_2_in_dead_state = False
        
        while not (terminated or truncated):
            if curr_symbol == "a":
                
                agent_id=1
                agent_1_row_num = row_nums[(agent_2_in_dead_state, agent_1_observation)]
                
                if agent_1_prev_row_num != -1 :
                    # Q-value update for agent 1
                    q_1[agent_1_prev_row_num][agent_1_communicate] += alpha * (reward_1 + gamma * np.max(q_1[agent_1_row_num]) - q_1[agent_1_prev_row_num][agent_1_communicate])
                    reward_1 = 0
                
                agent_1_communicate = get_action(q_1, agent_2_in_dead_state, agent_1_row_num, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                _, agent_1_observation, agent_2_observation = config
                
                agent_2_in_dead_state = agent_2_observation == -1
                    
                reward_1 += reward
                
                curr_symbol=info['input_alphabet']
                
                agent_1_prev_row_num = agent_1_row_num
                            
            if curr_symbol == "b":
                agent_id=2
                agent_2_row_num = row_nums[(agent_1_in_dead_state, agent_2_observation)]
                
                if agent_2_prev_row_num != -1:
                    # Q-value update for agent 2
                    q_2[agent_2_prev_row_num][agent_2_communicate] += alpha * (reward_2 + gamma * np.max(q_2[agent_2_row_num]) - q_2[agent_2_prev_row_num][agent_2_communicate])
                    reward_2 = 0
                
                agent_2_communicate = get_action(q_2, agent_1_in_dead_state, agent_2_row_num, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                
                _, agent_1_observation, agent_2_observation = config
                
                agent_1_in_dead_state = agent_1_observation == -1
                
                reward_2 += reward
                
                curr_symbol=info['input_alphabet']
                
                agent_2_prev_row_num = agent_2_row_num
        
        reward_1 += reward
        reward_2 += reward
        
        # Final Q-value updates
        q_1[agent_1_prev_row_num][agent_1_communicate] += alpha * (reward_1 + gamma * 0 - q_1[agent_1_prev_row_num][agent_1_communicate])
        q_2[agent_2_prev_row_num][agent_2_communicate] += alpha * (reward_2 + gamma * 0 - q_2[agent_2_prev_row_num][agent_2_communicate])

q_training_env = gym.make('ComplexEnv-v0', render_mode=None, string_mode="full")

q_1 = np.zeros((len(row_nums), q_training_env.action_space.n))
q_2 = np.zeros((len(row_nums), q_training_env.action_space.n))

q_training(q_training_env, q_1, q_2, epochs=1000000, alpha=0.01, gamma=0.5, epsilon=0.1)

q_1_df = pd.DataFrame(q_1, columns=["communicate", "do not communcate"])
q_2_df = pd.DataFrame(q_2, columns=["communicate", "do not communcate"])

q_1_df.to_csv(f'complex_problem/demo_q1_table.csv')
q_2_df.to_csv(f'complex_problem/demo_q2_table.csv')

# Training done, go to simulation.py for simulation