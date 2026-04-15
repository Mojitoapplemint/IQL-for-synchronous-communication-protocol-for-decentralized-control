import numpy as np
import gymnasium as gym
import pandas as pd
import random
import distributive_env as distributive_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = "ex_3_check_distributive_observability"


S_1 = {
    (0,  'a'): 0,
    (1,  'a'): 1,
    (2,  'a'): 2,
    (3,  'a'): 3,
    (4,  'a'): 4,
    (5,  'a'): 5,
    (6,  'a'): 6,
    (7,  'a'): 7,
    (8,  'a'): 8,
    (9,  'a'): 9,
    (-1, 'a'): 10,
    
    (0,  'd'): 11,
    (1,  'd'): 12,
    (2,  'd'): 13,
    (3,  'd'): 14,
    (4,  'd'): 15,
    (5,  'd'): 16,
    (6,  'd'): 17,
    (7,  'd'): 18,
    (8,  'd'): 19,
    (9,  'd'): 20,
    (-1, 'd'): 21,
    
}

S_2 = {
    (0,  'b'): 0,
    (1,  'b'): 1,
    (2,  'b'): 2,
    (3,  'b'): 3,
    (4,  'b'): 4,
    (5,  'b'): 5,
    (6,  'b'): 6,
    (7,  'b'): 7,
    (8,  'b'): 8,
    (9,  'b'): 9,
    (-1, 'b'): 10,
}


A1_OBS = ['a', 'd']

A2_OBS = ['b']


def get_action(q_table, row_num, epsilon):
    if random.uniform(0, 1) < epsilon:
        return np.argmin(q_table[row_num]) # Explore: choose the action that is not best
    else:
        return  np.argmax(q_table[row_num])  # Exploit: best action from Q-table

def q_training(env, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1, print_process=False):

    q_1 = np.zeros((len(S_1), env.action_space.n))
    q_2 = np.zeros((len(S_2), env.action_space.n))

    for episode in range(epochs):
        if (print_process and episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")
        
        config, info = env.reset()
        
        curr_event=info['curr_event']
        
        _, agent_1_belief, agent_2_belief = config
        
        s_1 = -1
        s_2 = -1
        
        terminated = False
        truncated = False
        
        agent_1_action = 0
        agent_2_action = 0
        
        reward_1 = 0
        reward_2 = 0     
        
        while not (terminated or truncated):
            if curr_event in A1_OBS:
                
                agent_id=1
                next_s_1 = S_1[(agent_1_belief, curr_event)]
                if s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[s_1][agent_1_action] += alpha * (reward_1 + gamma * np.max(q_1[next_s_1]) - q_1[s_1][agent_1_action])
                    reward_1 = 0
                
                agent_1_action = get_action(q_1, next_s_1, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_action))
                
                _, agent_1_belief, agent_2_belief = config
                
                
                comm_cost, penalty = reward
                
                reward_1 += comm_cost
                
                s_1 = next_s_1
                            
            if curr_event in A2_OBS:
                agent_id=2
                next_s_2 = S_2[(agent_2_belief, curr_event)]
                
                if s_2 != -1:
                    # Q-value update for agent 2
                    q_2[s_2][agent_2_action] += alpha * (reward_2 + gamma * np.max(q_2[next_s_2]) - q_2[s_2][agent_2_action])
                    reward_2 = 0
                
                agent_2_action = get_action(q_2, next_s_2, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_action))
                
                _, agent_1_belief, agent_2_belief = config
                
                comm_cost, penalty = reward
                
                reward_2 += comm_cost
                
                s_2 = next_s_2
                
            curr_event=info['curr_event']
        

        reward_2 += penalty
        reward_1 += penalty
        
        # Final Q-value updates
        q_1[s_1][agent_1_action] += alpha * (reward_1 + gamma * 0 - q_1[s_1][agent_1_action])
        q_2[s_2][agent_2_action] += alpha * (reward_2 + gamma * 0 - q_2[s_2][agent_2_action])

        # print(curr_symbol)
    return q_1, q_2
