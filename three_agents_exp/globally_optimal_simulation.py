import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_exp_env as three_agents_exp_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_exp_q import S_1, S_3, ACTIONS,A1_OBS, A3_OBS, get_action




env = gym.make('ThreeAgentsExpEnv-v0', render_mode=None, string_mode="simulation")

fail_count = 0
return_values = [0,0,0]

for i in range(4):
    state, info = env.reset()

    curr_event = info["curr_event"]
    # word = info["word"]

    _, agent_1_belief, agent_2_belief, agent_3_belief = state

    agent_1_in_dead_state, agent_2_in_dead_state, agent_3_in_dead_state = False, False, False

    terminated = False
    simulation_result = False

    count = [0,0,0]

    a1_return, a2_return, a3_return = 0, 0, 0

    while not terminated:
        if curr_event in A1_OBS:
            agent_id = 1        
            
            s_1 = S_1[(agent_1_belief,curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
            
            # Choosing action only based on the Q value; never explore
            if curr_event == 'a' and agent_1_belief == 1:
                a1_action = 3
            if curr_event == 'a' and agent_1_belief == 2:
                a1_action = 3
            elif curr_event == 'x' and agent_1_belief == 4:
                a1_action = 2
            elif curr_event == 'x':
                a1_action = 0
            
            a1_action = ACTIONS[a1_action]
            
            count[0] +=np.sum(a1_action)
            
            config, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
            
            comm_cost, penalty = reward
            
            a1_return += comm_cost
            
        # if curr_event == 'b':
        #     agent_id = 2
            
        #     s_2 = S_2[(agent_2_belief,curr_event, agent_1_in_dead_state, agent_3_in_dead_state)]
            
        #     # Choosing action only based on the Q value; never explore
        #     a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_2, epsilon=0)
            
        #     a2_action = ACTIONS[a2_action]
        #     count[1] +=np.sum(a2_action)
        #     config, reward, terminated, _, info = env.step((agent_id, a2_action))
            
        #     comm_cost, penalty = reward
                    
        #     a2_return += comm_cost
            
        if curr_event in A3_OBS:
            agent_id = 3
            
            s_3 = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
            
            # Choosing action only based on the Q value; never explore
            if curr_event == 'c' and agent_3_belief == 1:
                a3_action = 3
            if curr_event == 'c' and agent_3_belief == 3:
                a3_action = 3
            elif curr_event == 'y' and agent_3_belief == 5:
                a3_action = 1
            elif curr_event == 'y':
                a3_action = 0
            
            a3_action = ACTIONS[a3_action]
            count[2] +=np.sum(a3_action)
            
            config, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
            
            comm_cost, penalty = reward
            
            a3_return += comm_cost
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = config
        
        agent_1_in_dead_state = agent_1_belief == -1
        
        agent_2_in_dead_state = agent_2_belief == -1
        
        agent_3_in_dead_state = agent_3_belief == -1
        
        curr_event=info['curr_event']
        
        # print(a1_return, a2_return, a3_return)
        # print(count)

    if not simulation_result:
        fail_count += 1
        
    a1_return += penalty

    a2_return += penalty

    a3_return += penalty

    return_values[0] += a1_return
    return_values[1] += a2_return
    return_values[2] += a3_return
    # print()
    print(count)
    print(a1_return, a2_return, a3_return)

print(f"Failure: {fail_count}")
    
    
return_values = [return_values[i]/6 for i in range(3)]
return_values[0] = round(return_values[0], 2)
return_values[1] = round(return_values[1], 2)
return_values[2] = round(return_values[2], 2)

print(return_values)