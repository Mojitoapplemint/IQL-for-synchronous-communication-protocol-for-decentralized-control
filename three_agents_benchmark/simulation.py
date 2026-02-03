import numpy as np
import gymnasium as gym
import pandas as pd
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


S = {
    (1, False, False):0,
    (2, False, False):1,
    (3, False, False):2,
    (4, False, False):3,
    (5, False, False):4,
    (6, False, False):5,
    (7, False, False):6,
    (8, False, False):7,
    (9, False, False):8,
    (10,False, False):9,
    (11,False, False):10,
    (12,False, False):11,
    (13,False, False):12,
    (14,False, False):13,
    (15,False, False):14,
    (16,False, False):15,
    (17,False, False):16,
    (18,False, False):17,
    (19,False, False):18,
    (20,False, False):19,
    (21,False, False):20,
    (22,False, False):21,
    (-1,False, False):22,
    (1, True,  False):23,
    (2, True,  False):24,
    (3, True,  False):25,
    (4, True,  False):26,
    (5, True,  False):27,
    (6, True,  False):28,
    (7, True,  False):29,
    (8, True,  False):30,
    (9, True,  False):31,
    (10,True,  False):32,
    (11,True,  False):33,
    (12,True,  False):34,
    (13,True,  False):35,
    (14,True,  False):36,
    (15,True,  False):37,
    (16,True,  False):38,
    (17,True,  False):39,
    (18,True,  False):40,
    (19,True,  False):41,
    (20,True,  False):42,
    (21,True,  False):43,
    (22,True,  False):44,
    (-1,True,  False):45,
    (1, False, True):46,
    (2, False, True):47,
    (3, False, True):48,
    (4, False, True):49,
    (5, False, True):50,
    (6, False, True):51,
    (7, False, True):52,
    (8, False, True):53,
    (9, False, True):54,
    (10,False, True):55,
    (11,False, True):56,
    (12,False, True):57,
    (13,False, True):58,
    (14,False, True):59,
    (15,False, True):60,
    (16,False, True):61,
    (17,False, True):62,
    (18,False, True):63,
    (19,False, True):64,
    (20,False, True):65,
    (21,False, True):66,
    (22,False, True):67,
    (-1,False, True):68,
    (1, True,  True):69,
    (2, True,  True):70,
    (3, True,  True):71,
    (4, True,  True):72,
    (5, True,  True):73,
    (6, True,  True):74,
    (7, True,  True):75,
    (8, True,  True):76,
    (9, True,  True):77,
    (10,True,  True):78,
    (11,True,  True):79,
    (12,True,  True):80,
    (13,True,  True):81,
    (14,True,  True):82,
    (15,True,  True):83,
    (16,True,  True):84,
    (17,True,  True):85,
    (18,True,  True):86,
    (19,True,  True):87,
    (20,True,  True):88,
    (21,True,  True):89,
    (22,True,  True):90,
    (-1,True,  True):91,
}

ACTIONS = {
    0:[0,0],
    1:[0,1],
    2:[1,0],
    3:[1,1],
}

q_1 = pd.read_csv('three_agents_benchmark/three_agents_q1.csv', header=None).drop(0).to_numpy()
q_2 = pd.read_csv('three_agents_benchmark/three_agents_q2.csv', header=None).drop(0).to_numpy()
q_3 = pd.read_csv('three_agents_benchmark/three_agents_q3.csv', header=None).drop(0).to_numpy()

def get_action(q_table, agent_j_in_dead_state, agent_k_in_dead_state, row_num):
    
    # Both agents are in dead state, only action [0,0] is possible
    if agent_j_in_dead_state and agent_k_in_dead_state:
        return 0 
    
    # If one agent is in dead state, limit actions for that agent
    elif agent_j_in_dead_state or agent_k_in_dead_state: 
        if agent_j_in_dead_state:
            return np.argmax(q_table[row_num][[0,1]])  
        else:
            return np.argmax(q_table[row_num][[0,2]])  

    # Neither agent is in dead state, all actions possible
    return  np.argmax(q_table[row_num])

env = gym.make('ThreeAgentsEnv-v0', render_mode=None, string_mode="simulation")

count_list = []
string_list = []
returns_list = []

for i in range(6):
    state, info = env.reset()

    curr_event = info["curr_event"]
    string = info["string"]

    _, agent_1_obs, agent_2_obs, agent_3_obs = state

    agent_1_in_dead_state = False
    agent_2_in_dead_state = False
    agent_3_in_dead_state = False
    
    terminated = False
    simulation_result = False

    count = [0,0,0]

    a1_return = 0
    a2_return = 0
    a3_return = 0

    while not(terminated):
        if curr_event == 'a':
            agent_id = 1
            agent_1_row_num = S[(agent_1_obs, agent_2_in_dead_state, agent_3_in_dead_state)]
            
            a1_action = np.argmax(q_1[agent_1_row_num])
            # print(a1_action)
            
            # a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=agent_1_row_num)
            
            a1_action = ACTIONS[a1_action]
            
            # print(a1_action)
            
            count[0] += np.sum(a1_action)
            
            
            state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            agent_2_in_dead_state = agent_2_obs == -1
            agent_3_in_dead_state = agent_3_obs == -1
            
            curr_event = info["curr_event"]
            
            penalty, communication_cost = reward
                
            a1_return += communication_cost
            
        if curr_event == 'b':
            agent_id = 2
            agent_2_row_num = S[(agent_2_obs, agent_1_in_dead_state, agent_3_in_dead_state)]
            
            a2_action = np.argmax(q_2[agent_2_row_num])
            # a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=agent_2_row_num)
            
            a2_action = ACTIONS[a2_action]
            
            # print(a2_action)
            
            count[1] += np.sum(a2_action)
            
            
            state, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            penalty, communication_cost = reward
                
            a2_return += communication_cost
            
            agent_1_in_dead_state = agent_1_obs == -1
            agent_3_in_dead_state = agent_3_obs == -1
            
            curr_event = info["curr_event"]
        if curr_event == 'c':
            agent_id = 3
            agent_3_row_num = S[(agent_3_obs, agent_1_in_dead_state, agent_2_in_dead_state)]
            
            # a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=agent_3_row_num)
            
            a3_action = np.argmax(q_3[agent_3_row_num])
            
            a3_action = ACTIONS[a3_action]
            
            # print(a3_action)
            
            count[2] += np.sum(a3_action)
            
            state, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
            
            penalty, communication_cost = reward
                
            a3_return += communication_cost
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            agent_1_in_dead_state = agent_1_obs == -1
            agent_2_in_dead_state = agent_2_obs == -1
            
            curr_event = info["curr_event"]
    
    a1_return += penalty
    a2_return += penalty
    a3_return += penalty
    
    returns_list.append((a1_return, a2_return, a3_return))
    count_list.append(count)
    string_list.append(string)
    
for count, string, return_ in zip(count_list, string_list, returns_list):
    print(f"String: {string}, Communication Count: {count}, Returns: {return_}")