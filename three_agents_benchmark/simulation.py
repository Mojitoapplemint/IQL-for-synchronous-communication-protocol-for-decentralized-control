import numpy as np
import gymnasium as gym
import pandas as pd
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agent_q import S, ACTIONS

q_1 = pd.read_csv('three_agents_benchmark/three_agents_q1.csv', header=None).drop(0).to_numpy()
q_2 = pd.read_csv('three_agents_benchmark/three_agents_q2.csv', header=None).drop(0).to_numpy()
q_3 = pd.read_csv('three_agents_benchmark/three_agents_q3.csv', header=None).drop(0).to_numpy()

def get_action(q_table, agent_j_in_dead_state, agent_k_in_dead_state, row_num):
    
    # Both agents are in dead state, only action [0,0] is possible
    if agent_j_in_dead_state and agent_k_in_dead_state:
        return 0 
    
    # If one agent is in dead state, limit actions for that agent
    elif agent_j_in_dead_state or agent_k_in_dead_state: 
        if agent_j_in_dead_state:
            return np.argmax(q_table[row_num][[0,1]])  
        else:
            return np.argmax(q_table[row_num][[0,2]])  

    # Neither agent is in dead state, all actions possible
    return  np.argmax(q_table[row_num])

env = gym.make('ThreeAgentsEnv-v0', render_mode="human", string_mode="simulation")

count_list = []
string_list = []
returns_list = []

for i in range(6):
    state, info = env.reset()

    curr_event = info["curr_event"]
    string = info["string"]

    _, agent_1_obs, agent_2_obs, agent_3_obs = state

    agent_1_in_dead_state = False
    agent_2_in_dead_state = False
    agent_3_in_dead_state = False
    
    terminated = False
    simulation_result = False

    count = [0,0,0]

    a1_return = 0
    a2_return = 0
    a3_return = 0

    while not(terminated):
        if curr_event == 'a':
            agent_id = 1
            agent_1_row_num = S[(agent_1_obs, agent_2_in_dead_state, agent_3_in_dead_state)]
            
            a1_action = np.argmax(q_1[agent_1_row_num])
            # print(a1_action)
            
            # a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=agent_1_row_num)
            
            a1_action = ACTIONS[a1_action]
            
            # print(a1_action)
            
            count[0] += np.sum(a1_action)
            
            
            state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            agent_2_in_dead_state = agent_2_obs == -1
            agent_3_in_dead_state = agent_3_obs == -1
            
            curr_event = info["curr_event"]
            
            penalty, communication_cost = reward
                
            a1_return += communication_cost
            
        if curr_event == 'b':
            agent_id = 2
            agent_2_row_num = S[(agent_2_obs, agent_1_in_dead_state, agent_3_in_dead_state)]
            
            a2_action = np.argmax(q_2[agent_2_row_num])
            # a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=agent_2_row_num)
            
            a2_action = ACTIONS[a2_action]
            
            # print(a2_action)
            
            count[1] += np.sum(a2_action)
            
            
            state, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            penalty, communication_cost = reward
                
            a2_return += communication_cost
            
            agent_1_in_dead_state = agent_1_obs == -1
            agent_3_in_dead_state = agent_3_obs == -1
            
            curr_event = info["curr_event"]
        if curr_event == 'c':
            agent_id = 3
            agent_3_row_num = S[(agent_3_obs, agent_1_in_dead_state, agent_2_in_dead_state)]
            
            # a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=agent_3_row_num)
            
            a3_action = np.argmax(q_3[agent_3_row_num])
            
            a3_action = ACTIONS[a3_action]
            
            # print(a3_action)
            
            count[2] += np.sum(a3_action)
            
            state, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
            
            penalty, communication_cost = reward
                
            a3_return += communication_cost
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            agent_1_in_dead_state = agent_1_obs == -1
            agent_2_in_dead_state = agent_2_obs == -1
            
            curr_event = info["curr_event"]
    
    a1_return += penalty
    a2_return += penalty
    a3_return += penalty
    
    returns_list.append((a1_return, a2_return, a3_return))
    count_list.append(count)
    string_list.append(string)
    
for count, string, return_ in zip(count_list, string_list, returns_list):
    print(f"String: {string}, Communication Count: {count}, Returns: {return_}")