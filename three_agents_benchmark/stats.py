import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import sys
sys.path.insert(0, './problem_w_unobservable_events')
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agent_q import S, ACTIONS

successful_protocols = pd.read_csv('three_agents_benchmark/successful_protocols.csv')

returns_dict = {}

communication_counts = {
    "abcs$":[], 
    "bcas$":[], 
    "cabs$":[], 
    "acbs$":[], 
    "bacs$":[], 
    "cbas$":[]
}

a1_protocol_list = []
a2_protocol_list = []
a3_protocol_list = []


for index, row in successful_protocols.iterrows():
    # print(f"{index} / {len(successful_protocols)}", end="\r")
    protocol = row["Protocol"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    q_1 = protocol[0:92].copy()
    q_2 = protocol[92:184].copy()
    q_3 = protocol[184:276].copy()
    
    env = gym.make('ThreeAgentsEnv-v0', render_mode=None, string_mode="simulation")
    
    return_values = [0,0,0]
    
    a1_communication_protocol={
            1 :[0,0,0,0],
            2 :[0,0,0,0],
            3 :[0,0,0,0],
            4 :[0,0,0,0],
            5 :[0,0,0,0],
            6 :[0,0,0,0],
            7 :[0,0,0,0],
            8 :[0,0,0,0],
            9 :[0,0,0,0],
            10:[0,0,0,0],
            11:[0,0,0,0],
            12:[0,0,0,0],
            13:[0,0,0,0],
            14:[0,0,0,0],
            15:[0,0,0,0],
            16:[0,0,0,0],
            17:[0,0,0,0],
            19:[0,0,0,0],
            21:[0,0,0,0],
            22:[0,0,0,0],
            -1:[0,0,0,0],
    }
    
    a2_communication_protocol={
            1 :[0,0,0,0],
            2 :[0,0,0,0],
            3 :[0,0,0,0],
            4 :[0,0,0,0],
            5 :[0,0,0,0],
            6 :[0,0,0,0],
            7 :[0,0,0,0],
            8 :[0,0,0,0],
            9 :[0,0,0,0],
            10:[0,0,0,0],
            11:[0,0,0,0],
            12:[0,0,0,0],
            13:[0,0,0,0],
            14:[0,0,0,0],
            15:[0,0,0,0],
            16:[0,0,0,0],
            17:[0,0,0,0],
            19:[0,0,0,0],
            21:[0,0,0,0],
            22:[0,0,0,0],
            -1:[0,0,0,0],
    }
    
    a3_communication_protocol={
            1 :[0,0,0,0],
            2 :[0,0,0,0],
            3 :[0,0,0,0],
            4 :[0,0,0,0],
            5 :[0,0,0,0],
            6 :[0,0,0,0],
            7 :[0,0,0,0],
            8 :[0,0,0,0],
            9 :[0,0,0,0],
            10:[0,0,0,0],
            11:[0,0,0,0],
            12:[0,0,0,0],
            13:[0,0,0,0],
            14:[0,0,0,0],
            15:[0,0,0,0],
            16:[0,0,0,0],
            17:[0,0,0,0],
            19:[0,0,0,0],
            21:[0,0,0,0],
            22:[0,0,0,0],
            -1:[0,0,0,0],
    }
    
    for i in range(6):
        terminated = False
        simulation_result = False

        state, info = env.reset()
        input_word = info["string"]
        
        curr_event = info["curr_event"]
        
        _, a1_obs, a2_obs, a3_obs = state

        a1_in_dead_state = False
        a2_in_dead_state = False
        a3_in_dead_state = False
        
        a1_return = 0
        a2_return = 0
        a3_return = 0

        communication_count = [0,0,0]
        
        while not(terminated):
            if curr_event == 'a':
                agent_id = 1
                a1_row_num = S[(a1_obs, a2_in_dead_state, a3_in_dead_state)]
                
                a1_action = q_1[a1_row_num]
                
                a1_communication_protocol[a1_obs][a1_action] += 1
                
                a1_action = ACTIONS[a1_action]
                
                communication_count[0] += np.sum(a1_action)
                
                state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
                
                system_state, a1_obs, a2_obs, a3_obs = state
                
                a2_in_dead_state = a2_obs == -1
                a3_in_dead_state = a3_obs == -1
                
                curr_event = info["curr_event"]
                
                communication_cost, penalty = reward
                
                a1_return += communication_cost
                
            if curr_event == 'b':
                agent_id = 2
                a2_row_num = S[(a2_obs, a1_in_dead_state, a3_in_dead_state)]
                
                a2_action = q_2[a2_row_num]
                
                a2_communication_protocol[a2_obs][a2_action] += 1
                a2_action = ACTIONS[a2_action]
                
                communication_count[1] += np.sum(a2_action)
                
                state, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
                
                system_state, a1_obs, a2_obs, a3_obs = state
                
                a1_in_dead_state = a1_obs == -1
                a3_in_dead_state = a3_obs == -1
                
                curr_event = info["curr_event"]
                communication_cost, penalty = reward
                
                a2_return += communication_cost
                
            if curr_event == 'c':
                agent_id = 3
                a3_row_num = S[(a3_obs, a1_in_dead_state, a2_in_dead_state)]
                
                a3_action = q_3[a3_row_num]
                
                a3_communication_protocol[a3_obs][a3_action] += 1
                a3_action = ACTIONS[a3_action]
                
                communication_count[2] += np.sum(a3_action)
                
                state, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
                
                system_state, a1_obs, a2_obs, a3_obs = state
                
                a1_in_dead_state = a1_obs == -1
                a2_in_dead_state = a2_obs == -1
                
                curr_event = info["curr_event"]
                
                communication_cost, penalty = reward
                
                a3_return += communication_cost
        
            # print("reward:", reward)
        
        communication_counts.get(input_word).append(communication_count)
        
        if not simulation_result:
            print("\nError: Simulation failed unexpectedly.")
            break
        # print(a1_action, a2_action, a3_action)
        # print(a1_return, a2_return, a3_return)
        # print(communication_count)
        
        a1_return += penalty
        a2_return += penalty
        a3_return += penalty
        
        return_values[0] += a1_return
        return_values[1] += a2_return
        return_values[2] += a3_return
        
        
    return_values = [return_values[i]/6 for i in range(3)]
    return_values[0] = round(return_values[0], 2)
    return_values[1] = round(return_values[1], 2)
    return_values[2] = round(return_values[2], 2)
    
    if returns_dict.get(tuple(return_values)) is None:
        returns_dict[tuple(return_values)] = row["Counts"]
    else:
        returns_dict[tuple(return_values)] += row["Counts"]
    
    a1_protocol_list.append(a1_communication_protocol)
    a2_protocol_list.append(a2_communication_protocol)
    a3_protocol_list.append(a3_communication_protocol)

successful_protocols["Agent 1 Return"] = return_values[0]
successful_protocols["Agent 2 Return"] = return_values[1]
successful_protocols["Agent 3 Return"] = return_values[2]

print("\nAgent 1 Returns:", return_values[0])
print("Agent 2 Returns:", return_values[1])
print("Agent 3 Returns:", return_values[2])

for key in communication_counts:
    counts = communication_counts[key]
    avg_counts = [0,0,0]
    for count in counts:
        avg_counts[0] += count[0]
        avg_counts[1] += count[1]
        avg_counts[2] += count[2]
    avg_counts = [round(avg_counts[i]/len(counts),3) for i in range(3)]
    print(f"Average communication counts for input word '{key}': Agent 1: {avg_counts[0]}, Agent 2: {avg_counts[1]}, Agent 3: {avg_counts[2]}")

returns_dict_dist = {}
for key in returns_dict:
    return_sorted = np.sort(key)
    if returns_dict_dist.get(tuple(return_sorted)) is None:
        returns_dict_dist[tuple(return_sorted)] = returns_dict[key]
    else:
        returns_dict_dist[tuple(return_sorted)] += returns_dict[key]

returns_df = pd.DataFrame(list(returns_dict.items()), columns=["Returns (A1, A2, A3)", "Count"])
returns_df = returns_df.sort_values(by="Count", ascending=False)
print("\nReturns Distribution:")
print(returns_df)

returns_df = pd.DataFrame(list(returns_dict_dist.items()), columns=["Returns (A1, A2, A3)", "Count"])
returns_df = returns_df.sort_values(by="Count", ascending=False)
print("\nReturns Distribution:")
print(returns_df)

returns_df.to_csv("three_agents_benchmark/returns_distribution.csv", index=False)
