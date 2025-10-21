import numpy as np
import gymnasium as gym
import pandas as pd
import random
import sys
sys.path.insert(0, './complex_problem')
import complex_problem_env

agent_1_row_nums = {
    (0, 'a'):0,
    (2, 'a'):1,
    (3,'a'):2,
    (-1, 'a'):3,
    (0, '$'):4,
    (1, '$'):5,
    (2, '$'):6,
    (3, '$'):7,
    (4, '$'):8,
    (5, '$'):9,
    (-1, '$'):10
}

agent_2_row_nums = {
    (0, 'b'):0,
    (1, 'b'):1,
    (3,'b'):2,
    (-1, 'b'):3,
    (0, '$'):4,
    (1, '$'):5,
    (2, '$'):6,
    (3, '$'):7,
    (4, '$'):8,
    (5, '$'):9,
    (-1, '$'):10
}

class QAgent:
    def __init__(self, q_table, row_nums):
        self.q_table = q_table
        self.row_nums = row_nums
        self.current_row_num = -1
        self.action = None
        
        
    def get_action(self, observation, epsilon):
        self.current_row_num = self.row_nums[observation]
        
        if random.uniform(0, 1) < epsilon:
            self.action = np.argmin(self.q_table[self.current_row_num]) # Explore: choose the action that is not best
        else:
            self.action = np.argmax(self.q_table[self.current_row_num])  # Exploit: best action from Q-table

        return self.action
    

def get_action(q_table, row_num, epsilon):
    if random.uniform(0, 1) < epsilon:
        return np.argmin(q_table[row_num]) # Explore: choose the action that is not best
    else:
        return  np.argmax(q_table[row_num])  # Exploit: best action from Q-table
    


def q_training(env, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    
    q_1 = np.zeros((len(agent_1_row_nums), env.action_space.n))
    q_2 = np.zeros((len(agent_2_row_nums), env.action_space.n))

    for epoch in range(epochs):
        for epoch in range(epochs):
            if (epoch%100==0):
                print(str(100*epoch/epochs)+"%","done" , end="\r")
        
        config, info = env.reset()
        
        curr_symbol=info['input_alphabet']
        
        _, agent_1_observation, agent_2_observation = config
        
        agent_1_row_num = agent_1_row_nums[(agent_1_observation, curr_symbol)]
        agent_2_row_num = agent_2_row_nums[(agent_2_observation, curr_symbol)]
        
        agent_1_prev_row_num = agent_1_row_num
        agent_2_prev_row_num = agent_2_row_num
        
        terminated = False
        
        agent_communicate_1 = -1
        agent_communicate_2 = -1
        
        while not terminated:
            
            if curr_symbol == "a":
                agent_id=1
                agent_communicate_1 = get_action(q_1, agent_1_row_num, epsilon)
                config, reward, terminated, _, info = env.step((agent_id, agent_communicate_1))
                
                _, agent_1_observation, agent_2_observation = config
                
                curr_symbol=info['input_alphabet']
                
                agent_1_next_row_num = agent_1_row_nums[(agent_1_observation, curr_symbol)]
                
                q_1[agent_1_row_num][agent_communicate_1] += alpha * (reward + gamma * np.max(q_1[agent_1_next_row_num]) - q_1[agent_1_row_num][agent_communicate_1])

                if terminated:
                    q_2[agent_2_prev_row_num][agent_communicate_2] += alpha * (reward + gamma * np.max(q_2[agent_2_row_num]) - q_2[agent_2_prev_row_num][agent_communicate_2])

            
            if curr_symbol == "b":
                agent_id=2
                agent_communicate_2 = get_action(q_2, agent_2_row_num, epsilon)
                config, reward, terminated, _, info = env.step((agent_id, agent_communicate_2))
                
                _, agent_1_observation, agent_2_observation = config
                
                curr_symbol=info['input_alphabet']
                
                agent_2_next_row_num = agent_2_row_nums[(agent_2_observation, curr_symbol)]
                
                q_2[agent_2_row_num][agent_communicate_2] += alpha * (reward + gamma * np.max(q_2[agent_2_next_row_num]) - q_2[agent_2_row_num][agent_communicate_2])
                
                if terminated:
                    q_1[agent_1_prev_row_num][agent_communicate_1] += alpha * (reward + gamma * np.max(q_1[agent_1_row_num]) - q_1[agent_1_prev_row_num][agent_communicate_1])
        
                                    
            agent_1_prev_row_num = agent_1_row_num
            agent_2_prev_row_num = agent_2_row_num
        
            agent_1_row_num = agent_1_row_nums[(agent_1_observation, curr_symbol)]
            agent_2_row_num = agent_2_row_nums[(agent_2_observation, curr_symbol)]

    return q_1, q_2
     