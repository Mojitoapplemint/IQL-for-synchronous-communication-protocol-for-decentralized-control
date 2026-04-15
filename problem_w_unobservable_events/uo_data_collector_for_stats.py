import numpy as np
import gymnasium as gym
import pandas as pd
import uo_problem_env
from uo_problem_q import q_training, S_1, S_2, A1_OBS, A2_OBS, FOLDER_NAME


m_bottom={
    1:  {1,3},
    2:  {2,4,5,12,20,17},
    3:  {6,21},
    4:  {6,10},
    5:  {2,4,8,13,18},
    6:  {6},
    7:  {7},
    8:  {2,4,14},
    9:  {9},
    10: {10},
    11: {11},
    12: {2,4},
    13: {2,4,8,13,18},
    14: {14},
    15: {15},
    16: {16},
    17: {8,13,18},
    19: {19},
    21: {21},
    -1: {-1},
}

epochs=20000
alpha=0.01
gamma=0.1
epsilon=0.2
exp_number = 2

fail_rate_count={}
over_comm_rate_count={}
success_dict = {}
result_dict = {}
session_count = 1000

for i in range(session_count):
    print(str(100*i/session_count)+"%","done" , end="\r")
    env = gym.make('UOEnv-v1', render_mode=None, string_mode="training")
    q_1, q_2 = q_training(env, epochs=epochs, alpha=alpha, gamma=gamma, epsilon=epsilon)

    env = gym.make('UOEnv-v1', render_mode=None, string_mode="simulation")

    fail_count = 0
    over_comm_count = 0
    test_count = 300
    for _ in range (test_count):

        terminated = False
        simulation_result = False

        config, info = env.reset()
        
        word = info['word']
        
        # print(string)

        curr_event=info['curr_event']

        global_state, agent_1_belief, agent_2_belief = config

        while not (terminated):
            if curr_event in A1_OBS:
                
                agent_id=1
                agent_1_row_num = S_1[(agent_1_belief, curr_event)]

                agent_1_communicate = np.argmax(q_1[agent_1_row_num])
                    
                config, _, terminated, simulation_result, info = env.step((agent_id, agent_1_communicate))
                
                global_state, agent_1_belief, agent_2_belief = config
                

                
                curr_event=info['curr_event']
                            
            if curr_event in A2_OBS:
                agent_id=2
                agent_2_row_num = S_2[(agent_2_belief, curr_event)]
     
                agent_2_communicate = np.argmax(q_2[agent_2_row_num])

                config, _, terminated, simulation_result, info = env.step((agent_id, agent_2_communicate))
                
                global_state, agent_1_belief, agent_2_belief = config
                
                curr_event=info['curr_event']
        
        # print(global_state, agent_1_belief, agent_2_belief)
        if not simulation_result:
            fail_count += 1
        if global_state == agent_1_belief and global_state == agent_2_belief:
            over_comm_count += 1
            
        config = (global_state, agent_1_belief, agent_2_belief)
            
        if result_dict.get(config) is None:
            result_dict[config] = 1
        else:
            result_dict[config] = result_dict.get(config) + 1

    fail_rate = np.round(fail_count/test_count*100, 2)
    if fail_rate not in fail_rate_count:
        fail_rate_count[fail_rate] = 1
    else:
        fail_rate_count[fail_rate] += 1
    
    over_comm_rate = np.round(over_comm_count/test_count*100, 2)
    if over_comm_rate not in over_comm_rate_count:
        over_comm_rate_count[over_comm_rate] = 1
    else:
        over_comm_rate_count[over_comm_rate] += 1
    
    if fail_rate==0:
        q_1_comm_protocol = [0 for _ in range (40)]
        q_2_comm_protocol = [0 for _ in range (120)]
        for i in range(len(q_1_comm_protocol)):
            q_1_comm_protocol[i] = np.argmax(q_1[i])
        for i in range(len(q_2_comm_protocol)):
            q_2_comm_protocol[i] = np.argmax(q_2[i])
        
        protocol_key = (tuple(q_1_comm_protocol), tuple(q_2_comm_protocol))
        if protocol_key in success_dict:
            success_dict[protocol_key] += 1
        else:
            success_dict[protocol_key] = 1
    


# print result dictionary
for key in result_dict:
    print(f"<{key[0]}, {key[1]}, {key[2]}> => Count: {result_dict[key]}")


# Save results to CSV
fail_rate_df = pd.DataFrame(list(fail_rate_count.items()), columns=['Fail Rate (%)', 'Count'])
print(fail_rate_df)

over_comm_rate_df = pd.DataFrame(list(over_comm_rate_count.items()), columns=['Over-Communication Rate (%)', 'Count'])
print(over_comm_rate_df)

success_protocols_df = pd.DataFrame(list(success_dict.items()), columns=['Protocol', 'Success Count'])
success_protocols_df.to_csv(f"./{FOLDER_NAME}/successful_protocols_exp{exp_number}.csv", index=False)
