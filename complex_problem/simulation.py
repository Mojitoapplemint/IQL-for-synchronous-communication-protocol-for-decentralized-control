import numpy as np
import gymnasium as gym
import pandas as pd
import random
import complex_problem_env

q_1 = pd.read_csv("./complex_problem/demo_q1_table.csv")
q_2 = pd.read_csv("./complex_problem/demo_q2_table.csv")
q_1 = q_1.drop(q_1.columns[[0]], axis=1).to_numpy()
q_2 = q_2.drop(q_2.columns[[0]], axis=1).to_numpy()

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


env = gym.make("ComplexEnv-v0", render_mode = "simulation", string_mode="full")

terminated = False
truncated = False

config, info = env.reset()

_, agent_1_observation, agent_2_observation = config

curr_symbol=info['input_alphabet']

while not(terminated or truncated):
    if curr_symbol == "a":
        
        agent_id=1
        agent_1_row_num = agent_1_row_nums[(agent_1_observation, curr_symbol)]
        
        agent_communicate_1 = np.argmax(q_1[agent_1_row_num])
        
        config, reward, terminated, truncated, info = env.step((agent_id, agent_communicate_1))
        
        _, agent_1_observation, agent_2_observation = config
        
        curr_symbol=info['input_alphabet']
                    
    if curr_symbol == "b":
        agent_id=2
        agent_2_row_num = agent_2_row_nums[(agent_2_observation, curr_symbol)]
        
        agent_communicate_2 = np.argmax(q_2[agent_2_row_num])
        config, reward, terminated, truncated, info = env.step((agent_id, agent_communicate_2))
        
        _, agent_1_observation, agent_2_observation = config
        
        curr_symbol=info['input_alphabet']
        
