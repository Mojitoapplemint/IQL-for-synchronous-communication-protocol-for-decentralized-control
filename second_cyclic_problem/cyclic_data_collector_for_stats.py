import numpy as np
import gymnasium as gym
import pandas as pd
import random
import cyclic_problem_env
from cyclic_problem_q import q_training

PHI = {
    (False, 0 ):0,
    (False, 1 ):1,
    (False, 2 ):2,
    (False, 3 ):3,
    (False, 4 ):4,
    (False, 5 ):5,
    (False, 6 ):6,
    (False, 7 ):7,
    (False,-1 ):8,
    (True, 0 ):9,
    (True, 1 ):10,
    (True, 2 ):11,
    (True, 3 ):12,
    (True, 4 ):13,
    (True, 5 ):14,
    (True, 6 ):15,
    (True, 7 ):16,
    (True,-1 ):17,
}

fail_rate_count={}

success_dict = {}
fail_dict = {}
over_comm_rate_count={}
session_count = 1000

for i in range(session_count):
    print(str(100*i/session_count)+"%","done" , end="\r")
    
    fail_count = 0
    test_count = 100
    over_comm_count=0

    string_mode = "training"

    env = gym.make("CyclicEnv2-v0", render_mode = None, string_mode=string_mode)
    
    q_1, q_2 = q_training(env, epochs=2000, alpha=0.01, gamma=0.1, epsilon=0.1, print_process=False)


    string_mode = "simulation"

    env = gym.make("CyclicEnv2-v0", render_mode = None, string_mode=string_mode)

    for i in range (test_count):
        # print(i)

        terminated = False
        simulation_result = False

        config, info = env.reset()

        global_state, agent_1_belief, agent_2_belief = config

        curr_symbol=info['input_alphabet']
        
        string = info['string']

        agent_1_in_dead_state = False
        agent_2_in_dead_state = False

        while not(terminated):
            if curr_symbol == "a":
                
                agent_id=1
                agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
                
                if agent_2_in_dead_state:
                    agent_1_communicate = 0
                else:
                    agent_1_communicate = np.argmax(q_1[agent_1_row_num])
                
                config, reward, terminated, simulation_result, info = env.step((agent_id, agent_1_communicate))
                
                global_state, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                
                curr_symbol=info['input_alphabet']
                            
            if curr_symbol == "b":
                agent_id=2
                agent_2_row_num = PHI[(agent_1_in_dead_state, agent_2_belief)]
                
                if agent_1_in_dead_state:
                    agent_2_communicate = 0
                else:        
                    agent_2_communicate = np.argmax(q_2[agent_2_row_num])
                config, reward, terminated, simulation_result, info = env.step((agent_id, agent_2_communicate))
                
                global_state, agent_1_belief, agent_2_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                curr_symbol=info['input_alphabet']
        
        if not simulation_result:
            fail_count += 1
        if global_state == agent_1_belief and global_state == agent_2_belief:
            over_comm_count += 1
        
    fail_rate = np.round(fail_count/test_count*100, 2)
    
    q_1_comm_protocol = [0 for _ in range (17)]
    q_2_comm_protocol = [0 for _ in range (17)]
    for i in range(17):
        q_1_comm_protocol[i] = np.argmax(q_1[i])
        q_2_comm_protocol[i] = np.argmax(q_2[i])
    
    protocol_key = (tuple(q_1_comm_protocol), tuple(q_2_comm_protocol))
    if fail_rate==0:
        if protocol_key in success_dict:
            success_dict[protocol_key] += 1
        else:
            success_dict[protocol_key] = 1
    else:
        if protocol_key in fail_dict:
            fail_dict[protocol_key] += 1
        else:
            fail_dict[protocol_key] = 1
    
    if fail_rate in fail_rate_count:
        fail_rate_count[fail_rate] += 1
    else:
        fail_rate_count[fail_rate] = 1
        
    over_comm_rate = np.round(over_comm_count/test_count*100, 2)
    if over_comm_rate not in over_comm_rate_count:
        over_comm_rate_count[over_comm_rate] = 1
    else:
        over_comm_rate_count[over_comm_rate] += 1

fail_rate_count_df = pd.DataFrame(list(fail_rate_count.items()), columns=['Fail Rate (%)', 'Count'])
fail_rate_count_df = fail_rate_count_df.sort_values(by=['Fail Rate (%)'])
fail_rate_count_df.to_csv("./second_cyclic_problem/simulation_2_results.csv", index=False)

over_comm_rate_count_df = pd.DataFrame(list(over_comm_rate_count.items()), columns=['Over Communication Rate (%)', 'Count'])
over_comm_rate_count_df = over_comm_rate_count_df.sort_values(by=['Over Communication Rate (%)'])
over_comm_rate_count_df.to_csv("./second_cyclic_problem/simulation_2_over_communication_results.csv", index=False)


success_dict_df = pd.DataFrame(list(success_dict.items()), columns=['Communication Protocols', 'Success Count'])
success_dict_df.to_csv("./second_cyclic_problem/simulation_2_successful_protocols.csv", index=False)

fail_dict_df = pd.DataFrame(list(fail_dict.items()), columns=['Communication Protocols', 'Fail Count'])
fail_dict_df.to_csv("./second_cyclic_problem/simulation_2_failed_protocols.csv", index=False)

print("Fail Rate Count over", session_count, "sessions:")
print(fail_rate_count_df)

print("\nOver Communication Rate Count over", session_count, "sessions:")
print(over_comm_rate_count_df)

print(success_dict)

# Go to stats.py to analyze the results