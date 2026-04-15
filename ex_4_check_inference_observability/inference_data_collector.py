import numpy as np
import gymnasium as gym
import pandas as pd
import inference_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from inference_q import S_1, S_2, A1_OBS, A2_OBS, get_action, q_training, FOLDER_NAME

successful_protocol_dict = {}
failed_protocol_dict = {}
success_T_dict = {}
fail_T_dict = {}
session_count = 1000
for i in range(session_count):
    print(f"{i}/{session_count} done", end="\r")
    env = gym.make('InferenceEnv-v1', render_mode=None, string_mode="training")
    
    q_1, q_2 = q_training(env, epochs=20000, alpha=0.001, gamma=0.1, epsilon=0.1, print_process=False)
    
    env = gym.make('InferenceEnv-v1', render_mode=None, string_mode="simulation")
    
    fail_count = 0
    
    test_session = 100
    
    for _ in range(test_session):
        # print("here")
        terminated = False
        simulation_result = False

        v_state, info = env.reset()
        
        curr_event = info["curr_event"]
        
        _, agent_1_belief, agent_2_belief = v_state

        
        while not terminated:
            if curr_event in A1_OBS:
                agent_id = 1        
                
                s_1 = S_1[(agent_1_belief, curr_event)]
                
                # Choosing action only based on the Q value; never explore
                a1_action = get_action(q_1, row_num=s_1, epsilon=0)
                                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
                
            if curr_event in A2_OBS:
                agent_id = 2
                
                s_2 = S_2[(agent_2_belief, curr_event)]
                
                # Choosing action only based on the Q value; never explore
                a2_action = get_action(q_2, row_num=s_2, epsilon=0)
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
                
            _, agent_1_belief, agent_2_belief = v_state
                        
            curr_event=info['curr_event']
    
    
        if not simulation_result:
            fail_count += 1
    
            if fail_T_dict.get(tuple(v_state)) is None:
                fail_T_dict[tuple(v_state)] = 1
            else:
                fail_T_dict[tuple(v_state)] += 1
        else:
            if success_T_dict.get(tuple(v_state)) is None:
                success_T_dict[tuple(v_state)] = 1
            else:
                success_T_dict[tuple(v_state)] += 1


    a1_protocol = np.zeros((q_1.shape[0]), dtype=int)
    a2_protocol = np.zeros((q_2.shape[0]), dtype=int)
    
    for i in range(len(S_1)):
        a1_protocol[i]=np.argmax(q_1[i])

    
    for i in range(len(S_2)):
        a2_protocol[i]=np.argmax(q_2[i])

    
    protocol_key = (tuple(a1_protocol), tuple(a2_protocol))
    
    if fail_count == 0:
        if successful_protocol_dict.get(protocol_key) is None:
            successful_protocol_dict[protocol_key] = 1
        else:
            successful_protocol_dict[protocol_key] += 1
    else:
        if failed_protocol_dict.get(protocol_key) is None:
            failed_protocol_dict[protocol_key] = 1
        else:
            failed_protocol_dict[protocol_key] += 1

successful_protocols_df = pd.DataFrame(list(successful_protocol_dict.items()), columns=['Protocol', 'Counts'])

failed_protocol_df = pd.DataFrame(list(failed_protocol_dict.items()), columns=['Protocol', 'Counts'])

# print(successful_protocol_dict)

print(f"{np.sum(successful_protocols_df['Counts'])}/{session_count} sessions converged to a successful protocol.")

print("Terminal v states counts for successful protocols:")
for key in success_T_dict:
    print(f"<{key[0]}, {key[1]}, {key[2]}> => Count: {success_T_dict[key]}")

print("Terminal v states counts for failed protocols:")
for key in fail_T_dict:
    print(f"<{key[0]}, {key[1]}, {key[2]}> => Count: {fail_T_dict[key]}")

successful_protocols_df.to_csv(f'{FOLDER_NAME}/successful_protocols.csv', index=False)
failed_protocol_df.to_csv(f'{FOLDER_NAME}/failed_protocols.csv', index=False)