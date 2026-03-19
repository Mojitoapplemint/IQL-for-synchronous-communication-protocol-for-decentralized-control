import numpy as np
import gymnasium as gym
import pandas as pd
import random
import sys
sys.path.insert(0, './cyclic_problem_w_unobservable_events')
import cyclic_problem_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

PHI = {
    (False, 1 ):0,
    (False, 2 ):1,
    (False, 3 ):2,
    (False, 4 ):3,
    (False, 5 ):4,
    (False, 6 ):5,
    (False, 7 ):6,
    (False,-1 ):7,
    (True, 1 ):8,
    (True, 2 ):9,
    (True, 3 ):10,
    (True, 4 ):11,
    (True, 5 ):12,
    (True, 6 ):13,
    (True, 7 ):14,
    (True,-1 ):15,
}


def get_action(q_table, is_opponent_lost, row_num, epsilon):
    if is_opponent_lost:
        return 0
    if random.uniform(0, 1) < epsilon:
        return np.argmin(q_table[row_num]) # Explore: choose the action that is not best
    else:
        return  np.argmax(q_table[row_num])  # Exploit: best action from Q-table

def q_training(env, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1, print_process=False):

    q_1 = np.zeros((len(PHI), env.action_space.n))
    q_2 = np.zeros((len(PHI), env.action_space.n))

    for epoch in range(epochs):
        if (print_process and epoch%100==0):
            print(str(100*epoch/epochs)+"%","done" , end="\r")
        
        config, info = env.reset()
        
        curr_event=info['curr_event']
        
        _, agent_1_belief, agent_2_belief = config
        
        agent_1_prev_row_num = -1
        agent_2_prev_row_num = -1
        
        terminated = False
        truncated = False
        
        agent_1_communicate = 0
        agent_2_communicate = 0
        
        reward_1 = 0
        reward_2 = 0
        
        agent_1_in_dead_state = False
        agent_2_in_dead_state = False
        
        while not (terminated or truncated):
            if curr_event == "a":
                
                agent_id=1
                agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
                if agent_1_prev_row_num != -1 :
                    # Q-value update for agent 1
                    q_1[agent_1_prev_row_num][agent_1_communicate] += alpha * (reward_1 + gamma * np.max(q_1[agent_1_row_num]) - q_1[agent_1_prev_row_num][agent_1_communicate])
                    reward_1 = 0
                
                agent_1_communicate = get_action(q_1, agent_2_in_dead_state, agent_1_row_num, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                
                comm_cost, penalty = reward
                
                reward_1 += comm_cost
                
                curr_event=info['curr_event']
                
                agent_1_prev_row_num = agent_1_row_num
                            
            if curr_event == "b":
                agent_id=2
                agent_2_row_num = PHI[(agent_1_in_dead_state, agent_2_belief)]
                
                if agent_2_prev_row_num != -1:
                    # Q-value update for agent 2
                    q_2[agent_2_prev_row_num][agent_2_communicate] += alpha * (reward_2 + gamma * np.max(q_2[agent_2_row_num]) - q_2[agent_2_prev_row_num][agent_2_communicate])
                    reward_2 = 0
                
                agent_2_communicate = get_action(q_2, agent_1_in_dead_state, agent_2_row_num, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                comm_cost, penalty = reward
                
                reward_2 += comm_cost
                
                curr_event=info['curr_event']
                
                agent_2_prev_row_num = agent_2_row_num
        

        reward_2 += penalty
        reward_1 += penalty
        
        # Final Q-value updates
        q_1[agent_1_prev_row_num][agent_1_communicate] += alpha * (reward_1 + gamma * 0 - q_1[agent_1_prev_row_num][agent_1_communicate])
        q_2[agent_2_prev_row_num][agent_2_communicate] += alpha * (reward_2 + gamma * 0 - q_2[agent_2_prev_row_num][agent_2_communicate])

        # print(curr_symbol)
        
    return q_1, q_2



# Training done, go to simulation.py for simulation