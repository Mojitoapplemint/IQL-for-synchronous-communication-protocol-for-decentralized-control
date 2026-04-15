import numpy as np
import gymnasium as gym
import pandas as pd
import random
import inference_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = "ex_4_check_inference_observability"


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
    (10, 'a'):10,
    (11, 'a'):11,
    (12, 'a'):12,
    (13, 'a'):13,
    (14, 'a'):14,
    (15, 'a'):15,
    (16, 'a'):16,
    (17, 'a'):17,
    (18, 'a'):18,
    (19, 'a'):19,
    (20, 'a'):20,
    (21, 'a'):21,
    (-1, 'a'):22,
    
    (0,  'b'): 23,
    (1,  'b'): 24,
    (2,  'b'): 25,
    (3,  'b'): 26,
    (4,  'b'): 27,
    (5,  'b'): 28,
    (6,  'b'): 29,
    (7,  'b'): 30,
    (8,  'b'): 31,
    (9,  'b'): 32,
    (10, 'b'): 33,
    (11, 'b'): 34,
    (12, 'b'): 35,
    (13, 'b'): 36,
    (14, 'b'): 37,
    (15, 'b'): 38,
    (16, 'b'): 39,
    (17, 'b'): 40,
    (18, 'b'): 41,
    (19, 'b'): 42,
    (20, 'b'): 43,
    (21, 'b'): 44,
    (-1, 'b'): 45,
    
    (0,  'c'): 46,
    (1,  'c'): 47,
    (2,  'c'): 48,
    (3,  'c'): 49,
    (4,  'c'): 50,
    (5,  'c'): 51,
    (6,  'c'): 52,
    (7,  'c'): 53,
    (8,  'c'): 54,
    (9,  'c'): 55,
    (10, 'c'): 56,
    (11, 'c'): 57,
    (12, 'c'): 58,
    (13, 'c'): 59,
    (14, 'c'): 60,
    (15, 'c'): 61,
    (16, 'c'): 62,
    (17, 'c'): 63,
    (18, 'c'): 64,
    (19, 'c'): 65,
    (20, 'c'): 66,
    (21, 'c'): 67,
    (-1, 'c'): 68,
    
    (0,  'm'): 69,
    (1,  'm'): 70,
    (2,  'm'): 71,
    (3,  'm'): 72,
    (4,  'm'): 73,
    (5,  'm'): 74,
    (6,  'm'): 75,
    (7,  'm'): 76,
    (8,  'm'): 77,
    (9,  'm'): 78,
    (10, 'm'): 79,
    (11, 'm'): 80,
    (12, 'm'): 81,
    (13, 'm'): 82,
    (14, 'm'): 83,
    (15, 'm'): 84,
    (16, 'm'): 85,
    (17, 'm'): 86,
    (18, 'm'): 87,
    (19, 'm'): 88,
    (20, 'm'): 89,
    (21, 'm'): 90,
    (-1, 'm'): 91,
}

S_2 = {
    (0,  'p'): 0,
    (1,  'p'): 1,
    (2,  'p'): 2,
    (3,  'p'): 3,
    (4,  'p'): 4,
    (5,  'p'): 5,
    (6,  'p'): 6,
    (7,  'p'): 7,
    (8,  'p'): 8,
    (9,  'p'): 9,
    (10, 'p'):10,
    (11, 'p'):11,
    (12, 'p'):12,
    (13, 'p'):13,
    (14, 'p'):14,
    (15, 'p'):15,
    (16, 'p'):16,
    (17, 'p'):17,
    (18, 'p'):18,
    (19, 'p'):19,
    (20, 'p'):20,
    (21, 'p'):21,
    (-1, 'p'):22,
    
    (0,  'q'): 23,
    (1,  'q'): 24,
    (2,  'q'): 25,
    (3,  'q'): 26,
    (4,  'q'): 27,
    (5,  'q'): 28,
    (6,  'q'): 29,
    (7,  'q'): 30,
    (8,  'q'): 31,
    (9,  'q'): 32,
    (10, 'q'): 33,
    (11, 'q'): 34,
    (12, 'q'): 35,
    (13, 'q'): 36,
    (14, 'q'): 37,
    (15, 'q'): 38,
    (16, 'q'): 39,
    (17, 'q'): 40,
    (18, 'q'): 41,
    (19, 'q'): 42,
    (20, 'q'): 43,
    (21, 'q'): 44,
    (-1, 'q'): 45,
    
    (0,  'r'): 46,
    (1,  'r'): 47,
    (2,  'r'): 48,
    (3,  'r'): 49,
    (4,  'r'): 50,
    (5,  'r'): 51,
    (6,  'r'): 52,
    (7,  'r'): 53,
    (8,  'r'): 54,
    (9,  'r'): 55,
    (10, 'r'): 56,
    (11, 'r'): 57,
    (12, 'r'): 58,
    (13, 'r'): 59,
    (14, 'r'): 60,
    (15, 'r'): 61,
    (16, 'r'): 62,
    (17, 'r'): 63,
    (18, 'r'): 64,
    (19, 'r'): 65,
    (20, 'r'): 66,
    (21, 'r'): 67,
    (-1, 'r'): 68,
}


A1_OBS = ['a', 'b', 'c', 'm']

A2_OBS = ['p', 'q', 'r']

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
        
        agent_1_in_dead_state = False
        agent_2_in_dead_state = False
        
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
