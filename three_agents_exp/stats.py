import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import three_agents_exp_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_exp_q import S_1, S_3, ACTIONS,A1_OBS, A3_OBS, get_action, FOLDER_NAME

successful_protocols = pd.read_csv(f'{FOLDER_NAME}/successful_protocols.csv')
# successful_protocols = pd.read_csv('three_agents_benchmark/successful_protocols_with_returns.csv')
# successful_protocols = successful_protocols[successful_protocols["Agent 1 Return"]==0]

returns_dict = {}

communication_counts = {
    "caxs$":[], 
    "cays$":[], 
    "acys$":[], 
    "acxs$":[], 
}

return_value_list = []

for index, row in successful_protocols.iterrows():
    # print(f"{index} / {len(successful_protocols)}", end="\r")
    protocol = row["Protocol"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    q_1 = protocol[0:len(S_1)].copy()
    q_3 = protocol[len(S_1):].copy()
    
    
    env = gym.make('ThreeAgentsExpEnv-v0', render_mode=None, string_mode="simulation")
    
    return_values = [0,0,0]
    
    for i in range(4):
        terminated = False
        simulation_result = False

        v_state, info = env.reset()
        input_word = info["word"]
                
        curr_event = info["curr_event"]
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = v_state

        agent_1_in_dead_state = False
        agent_2_in_dead_state = False
        agent_3_in_dead_state = False
        
        a1_return = 0
        a2_return = 0
        a3_return = 0

        communication_count = [0,0,0]
        
        while not(terminated):
            
            if curr_event in A1_OBS:
                agent_id = 1        
                
                
                a1_row_num = S_1[(agent_1_belief,curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                a1_action = q_1[a1_row_num]
                a1_action = ACTIONS[a1_action]
                                
                communication_count[0] +=np.sum(a1_action)
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
                
                comm_cost, penalty = reward
                
                a1_return += comm_cost
                
            if curr_event in A3_OBS:
                agent_id = 3
                
                a3_row_num = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                a3_action = q_3[a3_row_num]
                a3_action = ACTIONS[a3_action]
                
                communication_count[2] += np.sum(a3_action)
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
                
                comm_cost, penalty = reward
                
                a3_return += comm_cost
            
            system_state, agent_1_belief, agent_2_belief, agent_3_belief = v_state    
                
            agent_1_in_dead_state = agent_1_belief == -1
            
            agent_2_in_dead_state = agent_2_belief == -1
            
            agent_3_in_dead_state = agent_3_belief == -1
            
            curr_event=info['curr_event']
        
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

successful_protocols.to_csv("three_agents_benchmark/successful_protocols_with_returns.csv", index=False)

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



