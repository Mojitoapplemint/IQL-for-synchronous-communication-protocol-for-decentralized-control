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
    (-1,False, False):16,
    (1, True,  False):17,
    (2, True,  False):18,
    (3, True,  False):19,
    (4, True,  False):20,
    (5, True,  False):21,
    (6, True,  False):22,
    (7, True,  False):23,
    (8, True,  False):24,
    (9, True,  False):25,
    (10,True,  False):26,
    (11,True,  False):27,
    (12,True,  False):28,
    (13,True,  False):29,
    (14,True,  False):30,
    (15,True,  False):31,
    (16,True,  False):32,
    (-1,True,  False):33,
    (1, False, True):34,
    (2, False, True):35,
    (3, False, True):36,
    (4, False, True):37,
    (5, False, True):38,
    (6, False, True):39,
    (7, False, True):40,
    (8, False, True):41,
    (9, False, True):42,
    (10,False, True):43,
    (11,False, True):44,
    (12,False, True):45,
    (13,False, True):46,
    (14,False, True):47,
    (15,False, True):48,
    (16,False, True):49,
    (-1,False, True):50,
    (1, True,  True):51,
    (2, True,  True):52,
    (3, True,  True):53,
    (4, True,  True):54,
    (5, True,  True):55,
    (6, True,  True):56,
    (7, True,  True):57,
    (8, True,  True):58,
    (9, True,  True):59,
    (10,True,  True):60,
    (11,True,  True):61,
    (12,True,  True):62,
    (13,True,  True):63,
    (14,True,  True):64,
    (15,True,  True):65,
    (16,True,  True):66,
    (-1,True,  True):67,
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
        return [0,0]  
    
    # If one agent is in dead state, limit actions for that agent
    elif agent_j_in_dead_state or agent_k_in_dead_state: 
        if agent_j_in_dead_state:
            return ACTIONS[np.argmax(q_table[row_num][[0,1]])]  
        else:
            return ACTIONS[np.argmax(q_table[row_num][[0,2]])]  
    
    # Neither agent is in dead state, all actions possible
    return  ACTIONS[np.argmax(q_table[row_num])]  



env = gym.make('ThreeAgentsEnv-v0', render_mode="simulation")

count_list = []

for i in range(6):
    state, info = env.reset()

    curr_event = info["curr_event"]

    _, agent_1_obs, agent_2_obs, agent_3_obs = state

    agent_1_in_dead_state = False
    agent_2_in_dead_state = False
    agent_3_in_dead_state = False
    
    terminated = False
    truncated = False

    count = 0

    while not(terminated or truncated):
        if curr_event == 'a':
            agent_id = 1
            agent_1_row_num = S[(agent_1_obs, agent_2_in_dead_state, agent_3_in_dead_state)]
            
            a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=agent_1_row_num)
            
            count += np.sum(a1_action)
            
            a1_action = [1,1,a1_action[0], a1_action[1]]
            
            state, _, terminated, truncated, info = env.step((agent_id, a1_action))
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            agent_2_in_dead_state = agent_2_obs == -1
            agent_3_in_dead_state = agent_3_obs == -1
            
            curr_event = info["curr_event"]
            
        if curr_event == 'b':
            agent_id = 2
            agent_2_row_num = S[(agent_2_obs, agent_1_in_dead_state, agent_3_in_dead_state)]
            
            a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=agent_2_row_num)
            
            count += np.sum(a2_action)
            
            a2_action = [1,1,a2_action[0], a2_action[1]]
            
            state, _, terminated, truncated, info = env.step((agent_id, a2_action))
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            agent_1_in_dead_state = agent_1_obs == -1
            agent_3_in_dead_state = agent_3_obs == -1
            
            curr_event = info["curr_event"]
        if curr_event == 'c':
            agent_id = 3
            agent_3_row_num = S[(agent_3_obs, agent_1_in_dead_state, agent_2_in_dead_state)]
            
            a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=agent_3_row_num)
            
            count += np.sum(a3_action)
            
            a3_action = [1,1,a3_action[0], a3_action[1]]
            
            state, _, terminated, truncated, info = env.step((agent_id, a3_action))
            
            _, agent_1_obs, agent_2_obs, agent_3_obs = state
            
            agent_1_in_dead_state = agent_1_obs == -1
            agent_2_in_dead_state = agent_2_obs == -1
            
            curr_event = info["curr_event"]
    
    count_list.append(count)
    
    
print(count_list)