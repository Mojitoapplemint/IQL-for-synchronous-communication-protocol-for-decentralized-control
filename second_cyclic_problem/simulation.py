import numpy as np
import gymnasium as gym
import pandas as pd
import random
import cyclic_problem_env

q_1 = pd.read_csv("./second_cyclic_problem/demo_q1_table.csv")
q_2 = pd.read_csv("./second_cyclic_problem/demo_q2_table.csv")
q_1 = q_1.drop(q_1.columns[[0]], axis=1).to_numpy()
q_2 = q_2.drop(q_2.columns[[0]], axis=1).to_numpy()

PHI = {
    (False, 0 ):0,
    (False, 1 ):1,
    (False, 2 ):2,
    (False, 3 ):3,
    (False, 4 ):4,
    (False, 5 ):5,
    (False, 6 ):6,
    (False, 7 ):7,
    (False,-1 ):8,
    (True, 0 ):9,
    (True, 1 ):10,
    (True, 2 ):11,
    (True, 3 ):12,
    (True, 4 ):13,
    (True, 5 ):14,
    (True, 6 ):15,
    (True, 7 ):16,
    (True,-1 ):17,
}

string_mode = "simulation" # options: "simulation", "half", "full"

env = gym.make("CyclicEnv2-v0", render_mode = "human", string_mode=string_mode)


terminated = False
simulation_result = False

config, info = env.reset()

global_state, agent_1_belief, agent_2_belief = config

curr_symbol=info['input_alphabet']

string = info['string']

agent_1_in_dead_state = False
agent_2_in_dead_state = False

while not(terminated):
    if curr_symbol == "a":
        
        agent_id=1
        agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
        
        if agent_2_in_dead_state:
            agent_1_communicate = 0
        else:
            agent_1_communicate = np.argmax(q_1[agent_1_row_num])
        
        config, reward, terminated, simulation_result, info = env.step((agent_id, agent_1_communicate))
        
        global_state, agent_1_belief, agent_2_belief = config
        
        agent_2_in_dead_state = agent_2_belief == -1
        
        curr_symbol=info['input_alphabet']
                    
    if curr_symbol == "b":
        agent_id=2
        agent_2_row_num = PHI[(agent_1_in_dead_state, agent_2_belief)]
        
        if agent_1_in_dead_state:
            agent_2_communicate = 0
        else:        
            agent_2_communicate = np.argmax(q_2[agent_2_row_num])
        config, reward, terminated, simulation_result, info = env.step((agent_id, agent_2_communicate))
        
        global_state, agent_1_belief, agent_2_belief = config
        
        agent_1_in_dead_state = agent_1_belief == -1
        
        curr_symbol=info['input_alphabet']
