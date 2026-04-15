import numpy as np
import gymnasium as gym
import pandas as pd
import random
import cyclic_problem_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = "cyclic_problem_w_unobservable_events"

S_1 = {
    (1 ,'a'):0,
    (2 ,'a'):1,
    (3 ,'a'):2,
    (4 ,'a'):3,
    (5 ,'a'):4,
    (6 ,'a'):5,
    (7 ,'a'):6,
    (-1 ,'a'):7,
}

S_2 = {
    (1 ,'b'):0,
    (2 ,'b'):1,
    (3 ,'b'):2,
    (4 ,'b'):3,
    (5 ,'b'):4,
    (6 ,'b'):5,
    (7 ,'b'):6,
    (-1 ,'b'):7,
}

A1_OBS = ['a']
A2_OBS = ['b']

def get_action(q_table, row_num, epsilon):
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
        
        agent_1_communicate = 0
        agent_2_communicate = 0
        
        reward_1 = 0
        reward_2 = 0
        
        while not (terminated or truncated):
            if curr_event in A1_OBS:
                
                agent_id=1
                next_s_1 = S_1[(agent_1_belief, curr_event)]
                if s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[s_1][agent_1_communicate] += alpha * (reward_1 + gamma * np.max(q_1[next_s_1]) - q_1[s_1][agent_1_communicate])
                    reward_1 = 0
                
                agent_1_communicate = get_action(q_1, next_s_1, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                
                comm_cost, penalty = reward
                
                reward_1 += comm_cost
                
                curr_event=info['curr_event']
                
                s_1 = next_s_1
                            
            if curr_event in A2_OBS:
                agent_id=2
                next_s_2 = S_2[(agent_2_belief, curr_event)]

                if s_2 != -1:
                    # Q-value update for agent 2
                    q_2[s_2][agent_2_communicate] += alpha * (reward_2 + gamma * np.max(q_2[next_s_2]) - q_2[s_2][agent_2_communicate])
                    reward_2 = 0
                
                agent_2_communicate = get_action(q_2, next_s_2, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                comm_cost, penalty = reward
                
                reward_2 += comm_cost
                
                curr_event=info['curr_event']
                
                s_2 = next_s_2
        

        reward_2 += penalty
        reward_1 += penalty
        
        # Final Q-value updates
        q_1[s_1][agent_1_communicate] += alpha * (reward_1 + gamma * 0 - q_1[s_1][agent_1_communicate])
        q_2[s_2][agent_2_communicate] += alpha * (reward_2 + gamma * 0 - q_2[s_2][agent_2_communicate])

        # print(curr_symbol)
        
    return q_1, q_2



# Training done, go to simulation.py for simulation