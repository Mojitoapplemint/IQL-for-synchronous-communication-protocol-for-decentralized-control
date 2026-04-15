import numpy as np
import gymnasium as gym
import pandas as pd
import random
import cyclic_problem_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from cyclic_problem_q import S_1, S_2, FOLDER_NAME, A1_OBS, A2_OBS


q_1 = [0, 0, 0, 0, 0, 0, 0, 0]
q_2 = [1, 0, 0, 0, 0, 0, 0, 0]


string_mode = "simulation" # options: "simulation", "training"

env = gym.make("CyclicEnv-v1", render_mode = "human", string_mode=string_mode, max_star=5)

success_count = 0
communicate = [0,0]

for i in range(5):
    terminated = False
    simulation_result = False

    config, info = env.reset()

    global_state, agent_1_belief, agent_2_belief = config

    curr_event=info['curr_event']

    word = info['word']

    agent_1_in_dead_state = False
    agent_2_in_dead_state = False

    cumulative_reward = [0,0]

    agent_1_prev_row_num = -1
    agent_2_prev_row_num = -1

    reward_1 = 0
    reward_2 = 0
    

    while not(terminated):
        if curr_event in A1_OBS:
            
            agent_id=1
            
            if agent_1_prev_row_num != -1 :
                # cumulative_reward[0] += (GAMMA**t_1)*reward_1
                cumulative_reward[0] += reward_1

                reward_1 = 0
            
            agent_1_row_num = S_1[(agent_1_belief, curr_event)]
            

            agent_1_communicate = q_1[agent_1_row_num]
            
            if agent_1_communicate ==1:
                communicate[0] += 1
            
            config, reward, terminated, simulation_result, info = env.step((agent_id, agent_1_communicate))
            
            global_state, agent_1_belief, agent_2_belief = config
            
            com_cost, penalty = reward
            
            reward_1 += com_cost
            
            agent_2_in_dead_state = agent_2_belief == -1
            
            curr_event=info['curr_event']
                        
        if curr_event in A2_OBS:
            agent_id=2
            
            if agent_2_prev_row_num != -1 :
                # cumulative_reward[0] += (GAMMA**t_1)*reward_1
                cumulative_reward[1] += reward_2

                reward_2 = 0
            
            agent_2_row_num = S_2[(agent_2_belief, curr_event)]
            
     
            agent_2_communicate = q_2[agent_2_row_num]
                
            if agent_2_communicate ==1:
                communicate[1] += 1
                
            config, reward, terminated, simulation_result, info = env.step((agent_id, agent_2_communicate))
            
            com_cost, penalty = reward
            
            global_state, agent_1_belief, agent_2_belief = config
        
            reward_2 += com_cost
            
            agent_1_in_dead_state = agent_1_belief == -1
            
            curr_event=info['curr_event']

    reward_2 += penalty
    reward_1 += penalty
    
    # cumulative_reward[0] += (GAMMA**t_1)*reward_1
    # cumulative_reward[1] += (GAMMA**t_2)*reward_2
    
    cumulative_reward[0] += reward_1
    cumulative_reward[1] += reward_2

    print(cumulative_reward)
    
    if simulation_result:
        success_count += 1
print(communicate)
print(f"Success count: {success_count}")
