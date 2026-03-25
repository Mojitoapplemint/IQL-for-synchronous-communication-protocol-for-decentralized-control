import numpy as np
import gymnasium as gym
import pandas as pd
import random
import distributive_env as distributive_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = "check_distributive_observability"


S_1 = {
    (0,  'a', False): 0,
    (1,  'a', False): 1,
    (2,  'a', False): 2,
    (3,  'a', False): 3,
    (4,  'a', False): 4,
    (5,  'a', False): 5,
    (6,  'a', False): 6,
    (7,  'a', False): 7,
    (8,  'a', False): 8,
    (9,  'a', False): 9,
    (-1, 'a', False): 10,
    
    (0,  'a', True): 11,
    (1,  'a', True): 12,
    (2,  'a', True): 13,
    (3,  'a', True): 14,
    (4,  'a', True): 15,
    (5,  'a', True): 16,
    (6,  'a', True): 17,
    (7,  'a', True): 18,
    (8,  'a', True): 19,
    (9,  'a', True): 20,
    (-1, 'a', True): 21,
    
    (0,  'd', False): 22,
    (1,  'd', False): 23,
    (2,  'd', False): 24,
    (3,  'd', False): 25,
    (4,  'd', False): 26,
    (5,  'd', False): 27,
    (6,  'd', False): 28,
    (7,  'd', False): 29,
    (8,  'd', False): 30,
    (9,  'd', False): 31,
    (-1, 'd', False): 32,
    
    (0,  'd', True): 33,
    (1,  'd', True): 34,
    (2,  'd', True): 35,
    (3,  'd', True): 36,
    (4,  'd', True): 37,
    (5,  'd', True): 38,
    (6,  'd', True): 39,
    (7,  'd', True): 40,
    (8,  'd', True): 41,
    (9,  'd', True): 42,
    (-1, 'd', True): 43,
    
}

S_2 = {
    (0,  'b', False): 0,
    (1,  'b', False): 1,
    (2,  'b', False): 2,
    (3,  'b', False): 3,
    (4,  'b', False): 4,
    (5,  'b', False): 5,
    (6,  'b', False): 6,
    (7,  'b', False): 7,
    (8,  'b', False): 8,
    (9,  'b', False): 9,
    (-1, 'b', False): 10,
    
    (0,  'b', True): 11,
    (1,  'b', True): 12,
    (2,  'b', True): 13,
    (3,  'b', True): 14,
    (4,  'b', True): 15,
    (5,  'b', True): 16,
    (6,  'b', True): 17,
    (7,  'b', True): 18,
    (8,  'b', True): 19,
    (9,  'b', True): 20,
    (-1, 'b', True): 21,
}


A1_OBS = ['a', 'd']

A2_OBS = ['b']


def get_action(q_table, is_opponent_dead, row_num, epsilon):
    if is_opponent_dead:
        return 0
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
        
        agent_1_in_dead_state = False
        agent_2_in_dead_state = False        
        
        while not (terminated or truncated):
            if curr_event in A1_OBS:
                
                agent_id=1
                next_s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state)]
                if s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[s_1][agent_1_action] += alpha * (reward_1 + gamma * np.max(q_1[next_s_1]) - q_1[s_1][agent_1_action])
                    reward_1 = 0
                
                agent_1_action = get_action(q_1, agent_2_in_dead_state, next_s_1, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_action))
                
                _, agent_1_belief, agent_2_belief = config
                
                
                comm_cost, penalty = reward
                
                reward_1 += comm_cost
                
                s_1 = next_s_1
                            
            if curr_event in A2_OBS:
                agent_id=2
                next_s_2 = S_2[(agent_2_belief, curr_event, agent_1_in_dead_state)]
                
                if s_2 != -1:
                    # Q-value update for agent 2
                    q_2[s_2][agent_2_action] += alpha * (reward_2 + gamma * np.max(q_2[next_s_2]) - q_2[s_2][agent_2_action])
                    reward_2 = 0
                
                agent_2_action = get_action(q_2, agent_1_in_dead_state, next_s_2, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_action))
                
                _, agent_1_belief, agent_2_belief = config
                
                comm_cost, penalty = reward
                
                reward_2 += comm_cost
                
                s_2 = next_s_2
                
            
            agent_1_in_dead_state = agent_1_belief == -1
                
            agent_2_in_dead_state = agent_2_belief == -1
                
            curr_event=info['curr_event']
        

        reward_2 += penalty
        reward_1 += penalty
        
        # Final Q-value updates
        q_1[s_1][agent_1_action] += alpha * (reward_1 + gamma * 0 - q_1[s_1][agent_1_action])
        q_2[s_2][agent_2_action] += alpha * (reward_2 + gamma * 0 - q_2[s_2][agent_2_action])

        # print(curr_symbol)
    return q_1, q_2
