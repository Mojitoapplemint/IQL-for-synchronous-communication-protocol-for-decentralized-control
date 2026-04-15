import numpy as np
import gymnasium as gym
import pandas as pd
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agent_q import q_training, ACTIONS, FOLDER_NAME

success_dict = {}
result_dict = {}
session_count = 100

fail_dict = {}

for i in range(session_count):
    print(str(100*i/session_count)+"%","done" , end="\r")
    
    env = gym.make('ThreeAgentsEnv-v1', render_mode=None, string_mode="training")
    
    q_1, q_2, q_3 = q_training(env, max_epochs=10000, alpha=0.001, gamma=0.1 , min_epsilon=0.1)
    
    env = gym.make('ThreeAgentsEnv-v1', render_mode=None, string_mode="simulation")
    
    fail_count = 0
    
    test_count = 6
    for _ in range (test_count):

        terminated = False
        simulation_result = False

        v_state, info = env.reset()
        
        curr_event = info["curr_event"]
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = v_state


        while not(terminated):
            if curr_event == 'a':
                agent_id = 1
                agent_1_row_num = agent_1_belief
                
                a1_action = np.argmax(q_1[agent_1_row_num])
                
                a1_action = ACTIONS[a1_action]
                
                v_state, _, terminated, simulation_result, info = env.step((agent_id, a1_action))

                
            if curr_event == 'b':
                agent_id = 2
                agent_2_row_num = agent_2_belief
                
                a2_action = np.argmax(q_2[agent_2_row_num])
                
                a2_action = ACTIONS[a2_action]
                
                v_state, _, terminated, simulation_result, info = env.step((agent_id, a2_action))

            if curr_event == 'c':
                agent_id = 3
                agent_3_row_num = agent_3_belief
                
                a3_action = np.argmax(q_3[agent_3_row_num])
                
                a3_action = ACTIONS[a3_action]
                
                v_state, _, terminated, simulation_result, info = env.step((agent_id, a3_action))
            
            system_state, agent_1_belief, agent_2_belief, agent_3_belief = v_state
                
            curr_event = info["curr_event"]

        if simulation_result == False:
            fail_count += 1
       
        if result_dict.get(tuple(v_state)) is None:
            result_dict[tuple(v_state)] = 1
        else:
            result_dict[tuple(v_state)] += 1
    
    a1_protocol = [np.argmax(q_1[i]) for i in range(q_1.shape[0])]
    a2_protocol = [np.argmax(q_2[i]) for i in range(q_2.shape[0])]
    a3_protocol = [np.argmax(q_3[i]) for i in range(q_3.shape[0])]
    
    protocol_key = (tuple(a1_protocol), tuple(a2_protocol), tuple(a3_protocol))
    if fail_count == 0:
        if success_dict.get(protocol_key) is None:
            success_dict[protocol_key] = 1
        else:
            success_dict[protocol_key] += 1
    else:
        if fail_dict.get(protocol_key) is None:
            fail_dict[protocol_key] = 1
        else:
            fail_dict[protocol_key] += 1

# print result dictionary
# for key in result_dict:
#     print(f"<{key[0]}, {key[1]}, {key[2]}, {key[3]}> => Count: {result_dict[key]}")

# Save successful protocols to CSV
successful_protocols_df = pd.DataFrame(list(success_dict.items()), columns=['Protocol', 'Counts'])
successful_protocols_df.to_csv(f'{FOLDER_NAME}/successful_protocols.csv', index=False)

# Save failed protocols to CSV
failed_protocols_df = pd.DataFrame(list(fail_dict.items()), columns=['Protocol', 'Counts'])
failed_protocols_df.to_csv(f'{FOLDER_NAME}/failed_protocols.csv', index=False)

print(f"\n{failed_protocols_df['Counts'].sum()}")