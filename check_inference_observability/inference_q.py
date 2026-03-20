import numpy as np
import gymnasium as gym
import pandas as pd
import random
import inference_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = "check_inference_observability"


S_1 = {
    (0,  'a', True): 0,
    (1,  'a', True): 1,
    (2,  'a', True): 2,
    (3,  'a', True): 3,
    (4,  'a', True): 4,
    (5,  'a', True): 5,
    (6,  'a', True): 6,
    (7,  'a', True): 7,
    (8,  'a', True): 8,
    (9,  'a', True): 9,
    (10, 'a', True):10,
    (11, 'a', True):11,
    (12, 'a', True):12,
    (13, 'a', True):13,
    (14, 'a', True):14,
    (15, 'a', True):15,
    (16, 'a', True):16,
    (17, 'a', True):17,
    (18, 'a', True):18,
    (19, 'a', True):19,
    (20, 'a', True):20,
    (21, 'a', True):21,
    (-1, 'a', True):22,
    
    (0,  'a', False): 23,
    (1,  'a', False): 24,
    (2,  'a', False): 25,
    (3,  'a', False): 26,
    (4,  'a', False): 27,
    (5,  'a', False): 28,
    (6,  'a', False): 29,
    (7,  'a', False): 30,
    (8,  'a', False): 31,
    (9,  'a', False): 32,
    (10, 'a', False): 33,
    (11, 'a', False): 34,
    (12, 'a', False): 35,
    (13, 'a', False): 36,
    (14, 'a', False): 37,
    (15, 'a', False): 38,
    (16, 'a', False): 39,
    (17, 'a', False): 40,
    (18, 'a', False): 41,
    (19, 'a', False): 42,
    (20, 'a', False): 43,
    (21, 'a', False): 44,
    (-1, 'a', False): 45,
    
    (0,  'b', True): 46,
    (1,  'b', True): 47,
    (2,  'b', True): 48,
    (3,  'b', True): 49,
    (4,  'b', True): 50,
    (5,  'b', True): 51,
    (6,  'b', True): 52,
    (7,  'b', True): 53,
    (8,  'b', True): 54,
    (9,  'b', True): 55,
    (10, 'b', True): 56,
    (11, 'b', True): 57,
    (12, 'b', True): 58,
    (13, 'b', True): 59,
    (14, 'b', True): 60,
    (15, 'b', True): 61,
    (16, 'b', True): 62,
    (17, 'b', True): 63,
    (18, 'b', True): 64,
    (19, 'b', True): 65,
    (20, 'b', True): 66,
    (21, 'b', True): 67,
    (-1, 'b', True): 68,
    
    (0,  'b', False): 69,
    (1,  'b', False): 70,
    (2,  'b', False): 71,
    (3,  'b', False): 72,
    (4,  'b', False): 73,
    (5,  'b', False): 74,
    (6,  'b', False): 75,
    (7,  'b', False): 76,
    (8,  'b', False): 77,
    (9,  'b', False): 78,
    (10, 'b', False): 79,
    (11, 'b', False): 80,
    (12, 'b', False): 81,
    (13, 'b', False): 82,
    (14, 'b', False): 83,
    (15, 'b', False): 84,
    (16, 'b', False): 85,
    (17, 'b', False): 86,
    (18, 'b', False): 87,
    (19, 'b', False): 88,
    (20, 'b', False): 89,
    (21, 'b', False): 90,
    (-1, 'b', False): 91,
    
    (0,  'c', True): 92,
    (1,  'c', True): 93,
    (2,  'c', True): 94,
    (3,  'c', True): 95,
    (4,  'c', True): 96,
    (5,  'c', True): 97,
    (6,  'c', True): 98,
    (7,  'c', True): 99,
    (8,  'c', True): 100,
    (9,  'c', True): 101,
    (10, 'c', True): 102,
    (11, 'c', True): 103,
    (12, 'c', True): 104,
    (13, 'c', True): 105,
    (14, 'c', True): 106,
    (15, 'c', True): 107,
    (16, 'c', True): 108,
    (17, 'c', True): 109,
    (18, 'c', True): 110,
    (19, 'c', True): 111,
    (20, 'c', True): 112,
    (21, 'c', True): 113,
    (-1, 'c', True): 114,

    (0,  'c', False): 115,
    (1,  'c', False): 116,
    (2,  'c', False): 117,
    (3,  'c', False): 118,
    (4,  'c', False): 119,
    (5,  'c', False): 120,
    (6,  'c', False): 121,
    (7,  'c', False): 122,
    (8,  'c', False): 123,
    (9,  'c', False): 124,
    (10, 'c', False): 125,
    (11, 'c', False): 126,
    (12, 'c', False): 127,
    (13, 'c', False): 128,
    (14, 'c', False): 129,
    (15, 'c', False): 130,
    (16, 'c', False): 131,
    (17, 'c', False): 132,
    (18, 'c', False): 133,
    (19, 'c', False): 134,
    (20, 'c', False): 135,
    (21, 'c', False): 136,
    (-1, 'c', False): 137,

    (0,  'm', True): 138,
    (1,  'm', True): 139,
    (2,  'm', True): 140,
    (3,  'm', True): 141,
    (4,  'm', True): 142,
    (5,  'm', True): 143,
    (6,  'm', True): 144,
    (7,  'm', True): 145,
    (8,  'm', True): 146,
    (9,  'm', True): 147,
    (10, 'm', True):148,
    (11, 'm', True):149,
    (12, 'm', True):150,
    (13, 'm', True):151,
    (14, 'm', True):152,
    (15, 'm', True):153,
    (16, 'm', True):154,
    (17, 'm', True):155,
    (18, 'm', True):156,
    (19, 'm', True):157,
    (20, 'm', True):158,
    (21, 'm', True):159,
    (-1, 'm', True):160,

    (0,  'm', False):161,
    (1,  'm', False):162,
    (2,  'm', False):163,
    (3,  'm', False):164,
    (4,  'm', False):165,
    (5,  'm', False):166,
    (6,  'm', False):167,
    (7,  'm', False):168,
    (8,  'm', False):169,
    (9,  'm', False):170,
    (10, 'm', False):171,
    (11, 'm', False):172,
    (12, 'm', False):173,
    (13, 'm', False):174,
    (14, 'm', False):175,
    (15, 'm', False):176,
    (16, 'm', False):177,
    (17, 'm', False):178,
    (18, 'm', False):179,
    (19, 'm', False):180,
    (20, 'm', False):181,
    (21, 'm', False):182,
    (-1, 'm', False):183
}

S_2 = {
    (0,  'p', True): 0,
    (1,  'p', True): 1,
    (2,  'p', True): 2,
    (3,  'p', True): 3,
    (4,  'p', True): 4,
    (5,  'p', True): 5,
    (6,  'p', True): 6,
    (7,  'p', True): 7,
    (8,  'p', True): 8,
    (9,  'p', True): 9,
    (10, 'p', True):10,
    (11, 'p', True):11,
    (12, 'p', True):12,
    (13, 'p', True):13,
    (14, 'p', True):14,
    (15, 'p', True):15,
    (16, 'p', True):16,
    (17, 'p', True):17,
    (18, 'p', True):18,
    (19, 'p', True):19,
    (20, 'p', True):20,
    (21, 'p', True):21,
    (-1, 'p', True):22,
    (0,  'p', False): 23,
    (1,  'p', False): 24,
    (2,  'p', False): 25,
    (3,  'p', False): 26,
    (4,  'p', False): 27,
    (5,  'p', False): 28,
    (6,  'p', False): 29,
    (7,  'p', False): 30,
    (8,  'p', False): 31,
    (9,  'p', False): 32,
    (10, 'p', False): 33,
    (11, 'p', False): 34,
    (12, 'p', False): 35,
    (13, 'p', False): 36,
    (14, 'p', False): 37,
    (15, 'p', False): 38,
    (16, 'p', False): 39,
    (17, 'p', False): 40,
    (18, 'p', False): 41,
    (19, 'p', False): 42,
    (20, 'p', False): 43,
    (21, 'p', False): 44,
    (-1, 'p', False): 45,
    
    (0,  'q', True): 46,
    (1,  'q', True): 47,
    (2,  'q', True): 48,
    (3,  'q', True): 49,
    (4,  'q', True): 50,
    (5,  'q', True): 51,
    (6,  'q', True): 52,
    (7,  'q', True): 53,
    (8,  'q', True): 54,
    (9,  'q', True): 55,
    (10, 'q', True): 56,
    (11, 'q', True): 57,
    (12, 'q', True): 58,
    (13, 'q', True): 59,
    (14, 'q', True): 60,
    (15, 'q', True): 61,
    (16, 'q', True): 62,
    (17, 'q', True): 63,
    (18, 'q', True): 64,
    (19, 'q', True): 65,
    (20, 'q', True): 66,
    (21, 'q', True): 67,
    (-1, 'q', True): 68,
    (0,  'q', False): 69,
    (1,  'q', False): 70,
    (2,  'q', False): 71,
    (3,  'q', False): 72,
    (4,  'q', False): 73,
    (5,  'q', False): 74,
    (6,  'q', False): 75,
    (7,  'q', False): 76,
    (8,  'q', False): 77,
    (9,  'q', False): 78,
    (10, 'q', False): 79,
    (11, 'q', False): 80,
    (12, 'q', False): 81,
    (13, 'q', False): 82,
    (14, 'q', False): 83,
    (15, 'q', False): 84,
    (16, 'q', False): 85,
    (17, 'q', False): 86,
    (18, 'q', False): 87,
    (19, 'q', False): 88,
    (20, 'q', False): 89,
    (21, 'q', False): 90,
    (-1, 'q', False): 91,
    
    (0,  'r', True): 92,
    (1,  'r', True): 93,
    (2,  'r', True): 94,
    (3,  'r', True): 95,
    (4,  'r', True): 96,
    (5,  'r', True): 97,
    (6,  'r', True): 98,
    (7,  'r', True): 99,
    (8,  'r', True): 100,
    (9,  'r', True): 101,
    (10, 'r', True): 102,
    (11, 'r', True): 103,
    (12, 'r', True): 104,
    (13, 'r', True): 105,
    (14, 'r', True): 106,
    (15, 'r', True): 107,
    (16, 'r', True): 108,
    (17, 'r', True): 109,
    (18, 'r', True): 110,
    (19, 'r', True): 111,
    (20, 'r', True): 112,
    (21, 'r', True): 113,
    (-1, 'r', True): 114,
    (0,  'r', False): 115,
    (1,  'r', False): 116,
    (2,  'r', False): 117,
    (3,  'r', False): 118,
    (4,  'r', False): 119,
    (5,  'r', False): 120,
    (6,  'r', False): 121,
    (7,  'r', False): 122,
    (8,  'r', False): 123,
    (9,  'r', False): 124,
    (10, 'r', False): 125,
    (11, 'r', False): 126,
    (12, 'r', False): 127,
    (13, 'r', False): 128,
    (14, 'r', False): 129,
    (15, 'r', False): 130,
    (16, 'r', False): 131,
    (17, 'r', False): 132,
    (18, 'r', False): 133,
    (19, 'r', False): 134,
    (20, 'r', False): 135,
    (21, 'r', False): 136,
    (-1, 'r', False): 137,
}


A1_OBS = ['a', 'b', 'c', 'm']

A2_OBS = ['p', 'q', 'r']

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

    for epoch in range(epochs):
        if (print_process and epoch%100==0):
            print(str(100*epoch/epochs)+"%","done" , end="\r")
        
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
