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

string_mode = "simulation" # options: "simulation", "training"

env = gym.make("CyclicEnv2-v0", render_mode = "human", string_mode=string_mode, max_star=5)

count = 0

for i in range(1):
    terminated = False
    simulation_result = False

    config, info = env.reset()

    global_state, agent_1_belief, agent_2_belief = config

    curr_symbol=info['input_alphabet']

    string = info['string']

    agent_1_in_dead_state = False
    agent_2_in_dead_state = False

    communicate = [0,0]

    while not(terminated):
        if curr_symbol == "a":
            
            agent_id=1
            agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
            
            if agent_2_in_dead_state:
                agent_1_communicate = 0
            else:
                agent_1_communicate = np.argmax(q_1[agent_1_row_num])
            
            if agent_1_communicate ==1:
                communicate[0] += 1
            
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
                
            if agent_2_communicate ==1:
                communicate[1] += 1
            config, reward, terminated, simulation_result, info = env.step((agent_id, agent_2_communicate))
            
            global_state, agent_1_belief, agent_2_belief = config
            
            agent_1_in_dead_state = agent_1_belief == -1
            
            curr_symbol=info['input_alphabet']

    if simulation_result:
        count += 1
print(communicate)
print(count)

q_1_comm_protocol = [0 for _ in range (len(PHI))]
q_2_comm_protocol = [0 for _ in range (len(PHI))]
for i in range(len(PHI)//2):
    q_1_comm_protocol[i] = np.argmax(q_1[i])
    q_2_comm_protocol[i] = np.argmax(q_2[i])

protocol_key = (tuple(q_1_comm_protocol), tuple(q_2_comm_protocol))

print(protocol_key)