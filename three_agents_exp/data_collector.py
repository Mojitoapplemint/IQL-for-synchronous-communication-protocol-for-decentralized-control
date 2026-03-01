import numpy as np
import gymnasium as gym
import pandas as pd
import three_agents_exp.three_agents_exp_env as three_agents_exp_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_exp.three_agents_exp_q import q_training, get_action, S_1, S_2, S_3, ACTIONS, FOLDER_NAME

successful_protocol_dict = {}
result_dict = {}
session_count = 10
for i in range(session_count):
    print(str(100*i/session_count)+"%","done" , end="\r")
    env = gym.make('ThreeAgentsComplexEnv-v0', render_mode=None, string_mode="training")
    
    q_1, q_2, q_3 = q_training(env, epochs=100000, alpha=0.001, gamma=0.9, min_epsilon=0.1)
    
    env = gym.make('ThreeAgentsComplexEnv-v0', render_mode=None, string_mode="simulation")
    
    fail_count = 0
    
    test_session = 100
    
    for _ in range(test_session):
        # print("here")
        terminated = False
        simulation_result = False

        v_state, info = env.reset()
        
        curr_event = info["curr_event"]
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = v_state

        agent_1_in_dead_state, agent_2_in_dead_state, agent_3_in_dead_state = False, False, False
        
        while not terminated:
            if curr_event == 'a':
                agent_id = 1        
                
                s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                # Choosing action only based on the Q value; never explore
                a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_1, epsilon=0)
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, ACTIONS[a1_action]))

                
            if curr_event == 'b':
                agent_id = 2
                
                s_2 = S_2[(agent_2_belief, curr_event, agent_1_in_dead_state, agent_3_in_dead_state)]
                
                # Choosing action only based on the Q value; never explore
                a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_2, epsilon=0)
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, ACTIONS[a2_action]))

                
            if curr_event == 'c':
                agent_id = 3
                
                s_3 = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                # Choosing action only based on the Q value; never explore
                a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=s_3, epsilon=0)
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, ACTIONS[a3_action]))


    
        if not simulation_result:
            fail_count += 1
    
        if result_dict.get(tuple(v_state)) is None:
            result_dict[tuple(v_state)] = 1
        else:
            result_dict[tuple(v_state)] += 1



    if fail_count == 0:
        a1_protocol = np.zeros((q_1.shape[0]))
        a2_protocol = np.zeros((q_2.shape[0]))
        a3_protocol = np.zeros((q_3.shape[0]))
        
        for i in range(q_1.shape[0]):
            if list(q_1[i]).count(0)!=4:
                a1_protocol[i]=np.argmax([item for item in q_1[i] if item !=0])
            else:
                a1_protocol[i]=0
        
        for i in range(q_2.shape[0]):
            if list(q_2[i]).count(0)!=4:
                a2_protocol[i]=np.argmax([item for item in q_2[i] if item !=0])
            else:
                a2_protocol[i]=0
        
        for i in range(q_3.shape[0]):
            if list(q_3[i]).count(0)!=4:
                a3_protocol[i]=np.argmax([item for item in q_3[i] if item !=0])
            else:
                a3_protocol[i]=0
        
        protocol_key = (tuple(a1_protocol), tuple(a2_protocol), tuple(a3_protocol))
        if successful_protocol_dict.get(protocol_key) is None:
            successful_protocol_dict[protocol_key] = 1
        else:
            successful_protocol_dict[protocol_key] += 1

successful_protocols_df = pd.DataFrame(list(successful_protocol_dict.items()), columns=['Protocol', 'Counts'])

print(successful_protocol_dict)

print(f"{np.sum(successful_protocols_df['Counts'])}/{session_count} sessions converged to a successful protocol.")

for key in result_dict:
    print(f"<{key[0]}, {key[1]}, {key[2]}, {key[3]}> => Count: {result_dict[key]}")

successful_protocols_df.to_csv(f'{FOLDER_NAME}/successful_protocols.csv', index=False)