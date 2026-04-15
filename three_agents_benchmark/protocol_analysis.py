import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agent_q import ACTIONS, FOLDER_NAME

successful_protocols = pd.read_csv(f'{FOLDER_NAME}/successful_protocols_with_returns.csv')

a1_idle = successful_protocols[successful_protocols["Agent 1 Return"] == 0]
a2_idle = successful_protocols[successful_protocols["Agent 2 Return"] == 0]
a3_idle = successful_protocols[successful_protocols["Agent 3 Return"] == 0]

A1_PROTOCOL={
    0:"Not Communicate",
    1:"Communicate to Agent 3",
    2:"Communicate to Agent 2",
    3:"Communicate to Both Agents"
}

A2_PROTOCOL={
    0:"Not Communicate",
    1:"Communicate to Agent 3",
    2:"Communicate to Agent 1",
    3:"Communicate to Both Agents"
}

A3_PROTOCOL={
    0:"Not Communicate",
    1:"Communicate to Agent 2",
    2:"Communicate to Agent 1",
    3:"Communicate to Both Agents"
}

for protocols, index in [(a1_idle, "Agent 1"), (a2_idle, "Agent 2"), (a3_idle, "Agent 3")]:    

    joint_protocol = {}
    for word in ['abcs$', 'bcas$', 'cabs$', 'acbs$', 'bacs$', 'cbas$']:
        joint_protocol[word] = (
            {i:[0,0,0,0] for i in [-1]+list(range(1,23))},  # Agent 1
            {i:[0,0,0,0] for i in [-1]+list(range(1,23))},  # Agent 2
            {i:[0,0,0,0] for i in [-1]+list(range(1,23))}   # Agent 3
        )
    
    for row in protocols["Protocol"]:
            
        protocol = row.replace("(","").replace(")","").split(", ")
        protocol = [int(x) for x in protocol]
        q_1 = protocol[0:23].copy()
        q_2 = protocol[23:46].copy()
        q_3 = protocol[46:69].copy()
        
        env = gym.make('ThreeAgentsEnv-v1', render_mode=None, string_mode="simulation")
        
        
        for i in range(6):
            
            
            terminated = False
            simulation_result = False

            state, info = env.reset()
            input_word = info["string"]
            
            curr_event = info["curr_event"]
            
            _, a1_obs, a2_obs, a3_obs = state

            a1_in_dead_state = False
            a2_in_dead_state = False
            a3_in_dead_state = False

            communication_count = [0,0,0]

            
            while not(terminated):
                if curr_event == 'a':
                    agent_id = 1
                    a1_row_num = a1_obs
                    
                    a1_action = q_1[a1_row_num]
                    
                    joint_protocol[input_word][0][a1_obs][a1_action] += 1
                    
                    a1_action = ACTIONS[a1_action]
                    
                    
                    state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
                    
                    system_state, a1_obs, a2_obs, a3_obs = state
                    
                    a2_in_dead_state = a2_obs == -1
                    a3_in_dead_state = a3_obs == -1
                    
                    curr_event = info["curr_event"]
                    
                    
                if curr_event == 'b':
                    agent_id = 2
                    a2_row_num = a2_obs
                    
                    a2_action = q_2[a2_row_num]
                    
                    joint_protocol[input_word][1][a2_obs][a2_action] += 1
        
                    a2_action = ACTIONS[a2_action]
                    
                    
                    state, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
                    
                    system_state, a1_obs, a2_obs, a3_obs = state
                    
                    a1_in_dead_state = a1_obs == -1
                    a3_in_dead_state = a3_obs == -1
                    
                    curr_event = info["curr_event"]

                    
                if curr_event == 'c':
                    agent_id = 3
                    a3_row_num = a3_obs
                    
                    a3_action = q_3[a3_row_num]
                    
                    joint_protocol[input_word][2][a3_obs][a3_action] += 1
                    
                    a3_action = ACTIONS[a3_action]
                    
                    state, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
                    
                    system_state, a1_obs, a2_obs, a3_obs = state
                    
                    a1_in_dead_state = a1_obs == -1
                    a2_in_dead_state = a2_obs == -1
                    
                    curr_event = info["curr_event"]

    
    for word in joint_protocol:
        for key in joint_protocol[word][0]:
            joint_protocol[word][0][key]=[x//len(protocols) for x in joint_protocol[word][0][key]]
            joint_protocol[word][1][key]=[x//len(protocols) for x in joint_protocol[word][1][key]]
            joint_protocol[word][2][key]=[x//len(protocols) for x in joint_protocol[word][2][key]]
    
    print(f"\n{index} Idle Protocols:\n")    
    for word in joint_protocol:
        print(f"Input Word: {word}")
        print("Agent 1 Protocol:")
        for key in joint_protocol[word][0]:
            comm = joint_protocol[word][0][key]
            if np.sum(comm) != 0:
                print(f"{key}: {A1_PROTOCOL[np.argmax(comm)]}")
                # print(f"{key}: {comm}")
        print("\nAgent 2 Protocol:")
        for key in joint_protocol[word][1]:
            comm = joint_protocol[word][1][key]
            if np.sum(comm) != 0:
                print(f"{key}: {A2_PROTOCOL[np.argmax(comm)]}")
                # print(f"{key}: {comm}")
        print("\nAgent 3 Protocol:")
        for key in joint_protocol[word][2]:
            comm = joint_protocol[word][2][key]
            if np.sum(comm) != 0:
                print(f"{key}: {A3_PROTOCOL[np.argmax(comm)]}")
                # print(f"{key}: {comm}")
        print("\n")

