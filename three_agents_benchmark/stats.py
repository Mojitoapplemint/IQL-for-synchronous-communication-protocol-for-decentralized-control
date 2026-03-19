import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agent_q import S, ACTIONS, FOLDER_NAME

successful_protocols = pd.read_csv(f'{FOLDER_NAME}/successful_protocols.csv')
# successful_protocols = pd.read_csv(f'{FOLDER_NAME}/successful_protocols_with_returns.csv')
# successful_protocols = successful_protocols[successful_protocols["Agent 1 Return"]==0]

returns_dict = {}

communication_counts = {
    "abcs$":[], 
    "bcas$":[], 
    "cabs$":[], 
    "acbs$":[], 
    "bacs$":[], 
    "cbas$":[]
}

return_value_list = []

for index, row in successful_protocols.iterrows():
    # print(f"{index} / {len(successful_protocols)}", end="\r")
    protocol = row["Protocol"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    q_1 = protocol[0:92].copy()
    q_2 = protocol[92:184].copy()
    q_3 = protocol[184:276].copy()
    
    env = gym.make('ThreeAgentsEnv-v0', render_mode=None, string_mode="simulation")
    
    return_values = [0,0,0]
    
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
    
    
    
    return_value_list.append(return_values)
    
    if returns_dict.get(tuple(return_values)) is None:
        returns_dict[tuple(return_values)] = row["Counts"]
    else:
        returns_dict[tuple(return_values)] += row["Counts"]
        
return_value_df = pd.DataFrame(return_value_list, columns=["Agent 1 Return", "Agent 2 Return", "Agent 3 Return"])
        
successful_protocols["Agent 1 Return"] = return_value_df["Agent 1 Return"]
successful_protocols["Agent 2 Return"] = return_value_df["Agent 2 Return"]
successful_protocols["Agent 3 Return"] = return_value_df["Agent 3 Return"]

successful_protocols.to_csv(f'{FOLDER_NAME}/successful_protocols_with_returns.csv', index=False)

for key in communication_counts:
    counts = communication_counts[key]
    avg_counts = [0,0,0]
    for count in counts:
        avg_counts[0] += count[0]
        avg_counts[1] += count[1]
        avg_counts[2] += count[2]
    avg_counts = [round(avg_counts[i]/len(counts),3) for i in range(3)]
    print(f"Average communication counts for input word '{key}': Agent 1: {avg_counts[0]}, Agent 2: {avg_counts[1]}, Agent 3: {avg_counts[2]}")


returns_df = pd.DataFrame(list(returns_dict.items()), columns=["Returns (A1, A2, A3)", "Count"])
returns_df = returns_df.sort_values(by="Count", ascending=False)
print("\nReturns Distribution:")
print(returns_df)



