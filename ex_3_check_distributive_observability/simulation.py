import numpy as np
import gymnasium as gym
import pandas as pd
import distributive_env as distributive_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from distributive_q import S_1, S_2, A1_OBS, A2_OBS, get_action, q_training, FOLDER_NAME


    
env = gym.make('DistributiveEnv-v1', render_mode="human", string_mode="simulation")

# Communicating 'a'
# q_1=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# q_2=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Communicating 'b'
q_1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
q_2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


fail_count = 0

test_session = 3

# print("here")
for i in range(test_session):
    terminated = False
    simulation_result = False

    v_state, info = env.reset()

    curr_event = info["curr_event"]

    _, agent_1_belief, agent_2_belief = v_state

    while not terminated:
        if curr_event in A1_OBS:
            agent_id = 1        
            
            s_1 = S_1[(agent_1_belief, curr_event)]

            a1_action = q_1[s_1]
                            
            v_state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
            
        if curr_event in A2_OBS:
            agent_id = 2
            
            s_2 = S_2[(agent_2_belief, curr_event)]
            

            a2_action = q_2[s_2]
            
            v_state, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
            
        _, agent_1_belief, agent_2_belief = v_state

        agent_1_in_dead_state = agent_1_belief == -1
        
        agent_2_in_dead_state = agent_2_belief == -1
                    
        curr_event=info['curr_event']
    
    if not simulation_result:
        fail_count += 1

print(f"Fail count: {fail_count}/{test_session}")