import numpy as np
import gymnasium as gym
import pandas as pd
import inference_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from inference_q import S_1, S_2, A1_OBS, A2_OBS, get_action, q_training, FOLDER_NAME


    
env = gym.make('InferenceEnv-v0', render_mode=None, string_mode="simulation")

q_1 = pd.read_csv(f'{FOLDER_NAME}/q_1.csv').to_numpy()
q_2 = pd.read_csv(f'{FOLDER_NAME}/q_2.csv').to_numpy()

q_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
q_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

fail_count = 0

# print("here")
for i in range(7):
    terminated = False
    simulation_result = False

    v_state, info = env.reset()

    curr_event = info["curr_event"]

    _, agent_1_belief, agent_2_belief = v_state

    agent_1_in_dead_state, agent_2_in_dead_state = False, False

    while not terminated:
        if curr_event in A1_OBS:
            agent_id = 1        
            
            s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state)]
            
            # a1_action = np.argmax(q_1[s_1])
            if agent_2_in_dead_state:
                a1_action = 0
            else:
                # if s_1 == 162:
                #     a1_action = 1
                # else:
                #     a1_action = 0
                
                a1_action = 0
                    
                # a1_action = q_1[s_1]
                            
            v_state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
            
        if curr_event in A2_OBS:
            agent_id = 2
            
            s_2 = S_2[(agent_2_belief, curr_event, agent_1_in_dead_state)]
            
            # a2_action = np.argmax(q_2[s_2])
            if agent_1_in_dead_state:
                a2_action = 0
            else:
                if s_2 == 23 or s_2 == 45 or s_2 == 69:
                    a2_action = 1
                else:
                    a2_action = 0
                # a2_action = q_2[s_2]
            
            v_state, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
            
        _, agent_1_belief, agent_2_belief = v_state

        agent_1_in_dead_state = agent_1_belief == -1
        
        agent_2_in_dead_state = agent_2_belief == -1
                    
        curr_event=info['curr_event']
    
    if not simulation_result:
        fail_count += 1

print(f"Fail count: {fail_count}/7")