import numpy as np
import gymnasium as gym
import pandas as pd
import uo_problem_env
from uo_problem_q import S_1, S_2, A1_OBS, A2_OBS, FOLDER_NAME

# q_1=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
# q_2= [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


q_1=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
q_2=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]




env = gym.make('UOEnv-v1', render_mode=None, string_mode="simulation")

for i in range(300):
    terminated = False
    truncated = False
    v_state, info = env.reset()

    curr_event=info['curr_event']
    word = info['word']

    system_state, agent_1_belief, agent_2_belief = v_state


    while not (terminated or truncated):
        if curr_event in A1_OBS:
            
            agent_id=1
            agent_1_row_num = S_1[(agent_1_belief, curr_event)]

            agent_1_communicate = q_1[agent_1_row_num]
            # agent_1_communicate = q_1[agent_1_row_num]
            
            if curr_event  == 'c' and agent_1_communicate == 1:
                print(f"system state: {system_state}, agent 1 belief: {agent_1_belief}, agent 2 belief: {agent_2_belief}")

                
            v_state, _, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
            
            system_state, agent_1_belief, agent_2_belief = v_state
            
            curr_event=info['curr_event']
                        
        if curr_event in A2_OBS:
            agent_id=2
            agent_2_row_num = S_2[(agent_2_belief, curr_event)]
    
            agent_2_communicate = q_2[agent_2_row_num]
            # agent_2_communicate = q_2[agent_2_row_num]
            
            # if curr_event  == 'c' and agent_2_communicate == 1:
            #     print(f"Simulation word: {word}, system state: {system_state}, agent 1 belief: {agent_1_belief}, agent 2 belief: {agent_2_belief}")

            v_state, _, terminated, _, info = env.step((agent_id, agent_2_communicate))
            
            system_state, agent_1_belief, agent_2_belief = v_state
            
            curr_event=info['curr_event']
