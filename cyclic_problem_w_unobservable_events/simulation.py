import numpy as np
import gymnasium as gym
import pandas as pd
import random
import cyclic_problem_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# q_1 = pd.read_csv("./second_cyclic_problem/demo_q1_table.csv")
# q_2 = pd.read_csv("./second_cyclic_problem/demo_q2_table.csv")

# q_1 = q_1.drop(q_1.columns[[0]], axis=1).to_numpy()
# q_2 = q_2.drop(q_2.columns[[0]], axis=1).to_numpy()

q_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
q_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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

env = gym.make("CyclicEnv-v0", render_mode = None, string_mode=string_mode, max_star=5)

count = 0
communicate = [0,0]

for i in range(1000):
    terminated = False
    simulation_result = False

    config, info = env.reset()

    global_state, agent_1_belief, agent_2_belief = config

    curr_symbol=info['input_alphabet']

    string = info['string']

    agent_1_in_dead_state = False
    agent_2_in_dead_state = False

    cumulative_reward = [0,0]

    agent_1_prev_row_num = -1
    agent_2_prev_row_num = -1

    reward_1 = 0
    reward_2 = 0
    

    while not(terminated):
        if curr_symbol == "a":
            
            agent_id=1
            
            if agent_1_prev_row_num != -1 :
                # cumulative_reward[0] += (GAMMA**t_1)*reward_1
                cumulative_reward[0] += reward_1

                reward_1 = 0
            
            agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
            
            if agent_2_in_dead_state:
                agent_1_communicate = 0
            else:
                # agent_1_communicate = np.argmax(q_1[agent_1_row_num])
                agent_1_communicate = q_1[agent_1_row_num]
            
            if agent_1_communicate ==1:
                communicate[0] += 1
            
            config, reward, terminated, simulation_result, info = env.step((agent_id, agent_1_communicate))
            
            global_state, agent_1_belief, agent_2_belief = config
            
            commun_cost, penalty = reward
            
            reward_1 += comm_cost
            
            agent_2_in_dead_state = agent_2_belief == -1
            
            curr_symbol=info['input_alphabet']
                        
        if curr_symbol == "b":
            agent_id=2
            
            if agent_2_prev_row_num != -1 :
                # cumulative_reward[0] += (GAMMA**t_1)*reward_1
                cumulative_reward[1] += reward_2

                reward_2 = 0
            
            agent_2_row_num = PHI[(agent_1_in_dead_state, agent_2_belief)]
            
            if agent_1_in_dead_state:
                agent_2_communicate = 0
            else:        
                # agent_2_communicate = np.argmax(q_2[agent_2_row_num])
                agent_2_communicate = q_2[agent_2_row_num]
                
            if agent_2_communicate ==1:
                communicate[1] += 1
            config, reward, terminated, simulation_result, info = env.step((agent_id, agent_2_communicate))
            
            comm_cost, penalty = reward
            
            global_state, agent_1_belief, agent_2_belief = config
        
            reward_2 += comm_cost
            
            agent_1_in_dead_state = agent_1_belief == -1
            
            curr_symbol=info['input_alphabet']

    reward_2 += penalty
    reward_1 += penalty
    
    # cumulative_reward[0] += (GAMMA**t_1)*reward_1
    # cumulative_reward[1] += (GAMMA**t_2)*reward_2
    
    cumulative_reward[0] += reward_1
    cumulative_reward[1] += reward_2

    print(cumulative_reward)
    
    if simulation_result:
        count += 1
print(communicate)
print(count)

# q_1_comm_protocol = [0 for _ in range (len(PHI))]
# q_2_comm_protocol = [0 for _ in range (len(PHI))]
# for i in range(len(PHI)//2):
#     q_1_comm_protocol[i] = np.argmax(q_1[i])
#     q_2_comm_protocol[i] = np.argmax(q_2[i])

# protocol_key = (tuple(q_1_comm_protocol), tuple(q_2_comm_protocol))

# print(protocol_key)