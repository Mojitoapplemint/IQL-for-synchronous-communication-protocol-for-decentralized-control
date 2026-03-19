import numpy as np
import gymnasium as gym
import pandas as pd
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agent_q import q_training, S, ACTIONS, FOLDER_NAME

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


success_dict = {}
result_dict = {}
session_count = 1000

for i in range(session_count):
    print(str(100*i/session_count)+"%","done" , end="\r")
    
    env = gym.make('ThreeAgentsEnv-v0', render_mode=None, string_mode="training")
    
    q_1, q_2, q_3 = q_training(env, max_epochs=10000, alpha=0.001, gamma=0.5, min_epsilon=0.1)
    
    env = gym.make('ThreeAgentsEnv-v0', render_mode=None, string_mode="simulation")
    
    fail_count = 0
    
    test_count = 6
    for _ in range (test_count):

        terminated = False
        simulation_result = False

        state, info = env.reset()
        
        curr_event = info["curr_event"]
        
        _, agent_1_obs, agent_2_obs, agent_3_obs = state

        agent_1_in_dead_state = False
        agent_2_in_dead_state = False
        agent_3_in_dead_state = False


        while not(terminated):
            if curr_event == 'a':
                agent_id = 1
                agent_1_row_num = S[(agent_1_obs, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=agent_1_row_num)
                
                a1_action = ACTIONS[a1_action]
                
                state, _, terminated, simulation_result, info = env.step((agent_id, a1_action))
                
                system_state, agent_1_obs, agent_2_obs, agent_3_obs = state
                
                agent_2_in_dead_state = agent_2_obs == -1
                agent_3_in_dead_state = agent_3_obs == -1
                
                curr_event = info["curr_event"]
                
            if curr_event == 'b':
                agent_id = 2
                agent_2_row_num = S[(agent_2_obs, agent_1_in_dead_state, agent_3_in_dead_state)]
                
                a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=agent_2_row_num)
                
                a2_action = ACTIONS[a2_action]
                
                state, _, terminated, simulation_result, info = env.step((agent_id, a2_action))
                
                system_state, agent_1_obs, agent_2_obs, agent_3_obs = state
                
                agent_1_in_dead_state = agent_1_obs == -1
                agent_3_in_dead_state = agent_3_obs == -1
                
                curr_event = info["curr_event"]
            if curr_event == 'c':
                agent_id = 3
                agent_3_row_num = S[(agent_3_obs, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=agent_3_row_num)
                
                a3_action = ACTIONS[a3_action]
                
                state, _, terminated, simulation_result, info = env.step((agent_id, a3_action))
                
                system_state, agent_1_obs, agent_2_obs, agent_3_obs = state
                
                agent_1_in_dead_state = agent_1_obs == -1
                agent_2_in_dead_state = agent_2_obs == -1
                
                curr_event = info["curr_event"]

        if simulation_result == False:
            fail_count += 1
       
        if result_dict.get(tuple(state)) is None:
            result_dict[tuple(state)] = 1
        else:
            result_dict[tuple(state)] += 1
    
    if fail_count == 0:
        a1_protocol = [np.argmax(q_1[i]) for i in range(q_1.shape[0])]
        a2_protocol = [np.argmax(q_2[i]) for i in range(q_2.shape[0])]
        a3_protocol = [np.argmax(q_3[i]) for i in range(q_3.shape[0])]
        
        protocol_key = (tuple(a1_protocol), tuple(a2_protocol), tuple(a3_protocol))
        if success_dict.get(protocol_key) is None:
            success_dict[protocol_key] = 1
        else:
            success_dict[protocol_key] += 1

print(fail_count)

# print result dictionary
for key in result_dict:
    print(f"<{key[0]}, {key[1]}, {key[2]}, {key[3]}> => Count: {result_dict[key]}")

# Save successful protocols to CSV
successful_protocols_df = pd.DataFrame(list(success_dict.items()), columns=['Protocol', 'Counts'])
successful_protocols_df.to_csv(f'{FOLDER_NAME}/successful_protocols.csv', index=False)

