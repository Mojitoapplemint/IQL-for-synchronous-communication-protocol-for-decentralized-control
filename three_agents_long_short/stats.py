import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import three_agents_ls_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_ls_q import S_1, S_3, ACTIONS,A1_OBS, A3_OBS, get_action, FOLDER_NAME


file_name = 'successful_protocols_exp_3'
# successful_protocols = pd.read_csv(f'{FOLDER_NAME}/failed_protocols_exp_3.csv')

successful_protocols = pd.read_csv(f'{FOLDER_NAME}/{file_name}.csv')
# successful_protocols = pd.read_csv('three_agents_benchmark/successful_protocols_with_returns.csv')
# successful_protocols = successful_protocols[successful_protocols["Agent 1 Return"]==0]

returns_dict = {}

communication_counts_x_in_5 = []
communication_counts_y_in_4 = []

communication_counts_a = []
communication_counts_c = []

T_state_dict_list = []

return_value_list = []

for index, row in successful_protocols.iterrows():
    # print(f"{index} / {len(successful_protocols)}", end="\r")
    protocol = row["Protocol"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    q_1 = protocol[0:len(S_1)].copy()
    q_3 = protocol[len(S_1):].copy()
    
    T_state_dict = {}
    communication_count_a = 0
    communication_count_c = 0
    
    communication_count_x_in_5 = 0
    communication_count_y_in_4 = 0
    
    env = gym.make('ThreeAgentsLSEnv-v0', render_mode=None, string_mode="simulation")
    
    
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
        
        while not(terminated):
            
            if curr_event in A1_OBS:
                agent_id = 1        
                
                a1_row_num = S_1[(agent_1_belief,curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                a1_action = q_1[a1_row_num]
                a1_action = ACTIONS[a1_action]
                
                if curr_event == 'a':            
                    communication_count_a +=np.sum(a1_action)
                if curr_event  == 'x' and agent_2_belief == 5:
                    communication_count_x_in_5 += np.sum(a1_action)
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
                
                comm_cost, penalty = reward
                
                a1_return += comm_cost
                
            if curr_event in A3_OBS:
                agent_id = 3
                
                a3_row_num = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                a3_action = q_3[a3_row_num]
                a3_action = ACTIONS[a3_action]
                
                if curr_event == 'c':
                    communication_count_c += np.sum(a3_action)
                if curr_event == 'y' and agent_2_belief == 4:
                    communication_count_y_in_4 += np.sum(a3_action)

                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
                
                comm_cost, penalty = reward
                
                a3_return += comm_cost
            
            system_state, agent_1_belief, agent_2_belief, agent_3_belief = v_state    
                
            agent_1_in_dead_state = agent_1_belief == -1
            
            agent_2_in_dead_state = agent_2_belief == -1
            
            agent_3_in_dead_state = agent_3_belief == -1
            
            curr_event=info['curr_event']
        
        
        # if not simulation_result:
        #     print("\nError: Simulation failed unexpectedly.")
        #     break

        
        a1_return += penalty
        a2_return += penalty
        a3_return += penalty
        
        return_values[0] += a1_return
        return_values[1] += a2_return
        return_values[2] += a3_return
        
        # if v_state not in T_state_dict:
        #     T_state_dict[v_state] = 1
        # else:
        #     T_state_dict[v_state] += 1
    
    communication_counts_a.append(communication_count_a)
    communication_counts_c.append(communication_count_c)
    communication_counts_x_in_5.append(communication_count_x_in_5)
    communication_counts_y_in_4.append(communication_count_y_in_4)
    
    T_state_dict_list.append(T_state_dict)
        
    return_values = [return_values[i]/4 for i in range(3)]
    return_values[0] = round(return_values[0], 2)
    return_values[1] = round(return_values[1], 2)
    return_values[2] = round(return_values[2], 2)
    
    return_value_list.append(return_values)
    
    if returns_dict.get(tuple(return_values)) is None:
        returns_dict[tuple(return_values)] = row["Counts"]
    else:
        returns_dict[tuple(return_values)] += row["Counts"]
        
return_value_df = pd.DataFrame(return_value_list, columns=["Agent 1 Return", "Agent 2 Return", "Agent 3 Return"])

communication_counts_a_df = pd.DataFrame(communication_counts_a, columns=["# comm for 'a'"])
communication_counts_c_df = pd.DataFrame(communication_counts_c, columns=["# comm for 'c'"])
communication_counts_x_in_5_df = pd.DataFrame(communication_counts_x_in_5, columns=["# comm for 'x' when system state in 5"])
communication_counts_y_in_4_df = pd.DataFrame(communication_counts_y_in_4, columns=["# comm for 'y' when system state in 4"])

T_state_dict_df = pd.DataFrame(T_state_dict_list, columns = T_state_dict_list[0].keys())

# print(T_state_dict_df)

successful_protocols["Agent 1 Return"] = return_value_df["Agent 1 Return"]
successful_protocols["Agent 2 Return"] = return_value_df["Agent 2 Return"]
successful_protocols["Agent 3 Return"] = return_value_df["Agent 3 Return"]

successful_protocols.to_csv(f"{FOLDER_NAME}/{file_name}_with_returns.csv", index=False)



returns_df = pd.DataFrame(list(returns_dict.items()), columns=["Returns (A1, A2, A3)", "Count"])
returns_df = returns_df.sort_values(by="Count", ascending=False)
print("\nReturns Distribution:")
print(returns_df)



