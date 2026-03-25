import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_ls_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


FOLDER_NAME = 'three_agents_long_short'

S_1 = {
    (1,'a', False, False):1,
    (2,'a', False, False):2,
    (3,'a', False, False):3,
    (4,'a', False, False):4,
    (5,'a', False, False):5,
    (6,'a', False, False):6,
    (7,'a', False, False):7,
    (8,'a', False, False):8,
    (9,'a', False, False):9,
    (10,'a', False, False):10,
    (11,'a', False, False):11,
    (12,'a', False, False):12,
    (13,'a', False, False):13,
    (-1,'a', False, False):14,
    (1,'a', True, False):15,
    (2,'a', True, False):16,
    (3,'a', True, False):17,
    (4,'a', True, False):18,
    (5,'a', True, False):19,
    (6,'a', True, False):20,
    (7,'a', True, False):21,
    (8,'a', True, False):22,
    (9,'a', True, False):23,
    (10,'a', True, False):24,
    (11,'a', True, False):25,
    (12,'a', True, False):26,
    (13,'a', True, False):27,
    (-1,'a', True, False):28,
    (1,'a',  False ,True ):29,
    (2,'a',  False ,True ):30,
    (3,'a',  False ,True ):31,
    (4,'a',  False ,True ):32,
    (5,'a',  False ,True ):33,
    (6,'a',  False ,True ):34,
    (7,'a',  False ,True ):35,
    (8,'a',  False ,True ):36,
    (9,'a',  False ,True ):37,
    (10,'a',  False ,True ):38,
    (11,'a',  False ,True ):39,
    (12,'a',  False ,True ):40,
    (13,'a',  False ,True ):41,
    (-1,'a',  False ,True ):42,
    (1,'a',  True ,True ):43,
    (2,'a',  True ,True ):44,
    (3,'a',  True ,True ):45,
    (4,'a',  True ,True ):46,
    (5,'a',  True ,True ):47,
    (6,'a',  True ,True ):48,
    (7,'a',  True ,True ):49,
    (8,'a',  True ,True ):50,
    (9,'a',  True ,True ):51,
    (10,'a',  True ,True ):52,
    (11,'a',  True ,True ):53,
    (12,'a',  True ,True ):54,
    (13,'a',  True ,True ):55,
    (-1,'a',  True ,True ):56,
    
    (1,'x', False, False):57,
    (2,'x', False, False):58,
    (3,'x', False, False):59,
    (4,'x', False, False):60,
    (5,'x', False, False):61,
    (6,'x', False, False):62,
    (7,'x', False, False):63,
    (8,'x', False, False):64,
    (9,'x', False, False):65,
    (10,'x', False, False):66,
    (11,'x', False, False):67,
    (12,'x', False, False):68,
    (13,'x', False, False):69,
    (-1,'x', False, False):70,
    (1,'x', True, False):71,
    (2,'x', True, False):72,
    (3,'x', True, False):73,
    (4,'x', True, False):74,
    (5,'x', True, False):75,
    (6,'x', True, False):76,
    (7,'x', True, False):77,
    (8,'x', True, False):78,
    (9,'x', True, False):79,
    (10,'x', True, False):80,
    (11,'x', True, False):81,
    (12,'x', True, False):82,
    (13,'x', True, False):83,
    (-1,'x', True, False):84,
    (1,'x',  False ,True ):85,
    (2,'x',  False ,True ):86,
    (3,'x',  False ,True ):87,
    (4,'x',  False ,True ):88,
    (5,'x',  False ,True ):89,
    (6,'x',  False ,True ):90,
    (7,'x',  False ,True ):91,
    (8,'x',  False ,True ):92,
    (9,'x',  False ,True ):93,
    (10,'x',  False ,True ):94,
    (11,'x',  False ,True ):95,
    (12,'x',  False ,True ):96,
    (13,'x',  False ,True ):97,
    (-1,'x',  False ,True ):98,
    (1,'x',  True ,True ):99,
    (2,'x',  True ,True ):100,
    (3,'x',  True ,True ):101,
    (4,'x',  True ,True ):102,
    (5,'x',  True ,True ):103,
    (6,'x',  True ,True ):104,
    (7,'x',  True ,True ):105,
    (8,'x',  True ,True ):106,
    (9,'x',  True ,True ):107,
    (10,'x',  True ,True ):108,
    (11,'x',  True ,True ):109,
    (12,'x',  True ,True ):110,
    (13,'x',  True ,True ):111,
    (-1,'x',  True ,True ):112
}

S_3 = {
    (1,'c', False, False):1,
    (2,'c', False, False):2,
    (3,'c', False, False):3,
    (4,'c', False, False):4,
    (5,'c', False, False):5,
    (6,'c', False, False):6,
    (7,'c', False, False):7,
    (8,'c', False, False):8,
    (9,'c', False, False):9,
    (10,'c', False, False):10,
    (11,'c', False, False):11,
    (12,'c', False, False):12,
    (13,'c', False, False):13,
    (-1,'c', False, False):14,
    (1,'c', True, False):15,
    (2,'c', True, False):16,
    (3,'c', True, False):17,
    (4,'c', True, False):18,
    (5,'c', True, False):19,
    (6,'c', True, False):20,
    (7,'c', True, False):21,
    (8,'c', True, False):22,
    (9,'c', True, False):23,
    (10,'c', True, False):24,
    (11,'c', True, False):25,
    (12,'c', True, False):26,
    (13,'c', True, False):27,
    (-1,'c', True, False):28,
    (1,'c',  False ,True ):29,
    (2,'c',  False ,True ):30,
    (3,'c',  False ,True ):31,
    (4,'c',  False ,True ):32,
    (5,'c',  False ,True ):33,
    (6,'c',  False ,True ):34,
    (7,'c',  False ,True ):35,
    (8,'c',  False ,True ):36,
    (9,'c',  False ,True ):37,
    (10,'c',  False ,True ):38,
    (11,'c',  False ,True ):39,
    (12,'c',  False ,True ):40,
    (13,'c',  False ,True ):41,
    (-1,'c',  False ,True ):42,
    (1,'c',  True ,True ):43,
    (2,'c',  True ,True ):44,
    (3,'c',  True ,True ):45,
    (4,'c',  True ,True ):46,
    (5,'c',  True ,True ):47,
    (6,'c',  True ,True ):48,
    (7,'c',  True ,True ):49,
    (8,'c',  True ,True ):50,
    (9,'c',  True ,True ):51,
    (10,'c',  True ,True ):52,
    (11,'c',  True ,True ):53,
    (12,'c',  True ,True ):54,
    (13,'c',  True ,True ):55,
    (-1,'c',  True ,True ):56,
    
    (1,'y', False, False):57,
    (2,'y', False, False):58,
    (3,'y', False, False):59,
    (4,'y', False, False):60,
    (5,'y', False, False):61,
    (6,'y', False, False):62,
    (7,'y', False, False):63,
    (8,'y', False, False):64,
    (9,'y', False, False):65,
    (10,'y', False, False):66,
    (11,'y', False, False):67,
    (12,'y', False, False):68,
    (13,'y', False, False):69,
    (-1,'y', False, False):70,
    (1,'y', True, False):71,
    (2,'y', True, False):72,
    (3,'y', True, False):73,
    (4,'y', True, False):74,
    (5,'y', True, False):75,
    (6,'y', True, False):76,
    (7,'y', True, False):77,
    (8,'y', True, False):78,
    (9,'y', True, False):79,
    (10,'y', True, False):80,
    (11,'y', True, False):81,
    (12,'y', True, False):82,
    (13,'y', True, False):83,
    (-1,'y', True, False):84,
    (1,'y',  False ,True ):85,
    (2,'y',  False ,True ):86,
    (3,'y',  False ,True ):87,
    (4,'y',  False ,True ):88,
    (5,'y',  False ,True ):89,
    (6,'y',  False ,True ):90,
    (7,'y',  False ,True ):91,
    (8,'y',  False ,True ):92,
    (9,'y',  False ,True ):93,
    (10,'y',  False ,True ):94,
    (11,'y',  False ,True ):95,
    (12,'y',  False ,True ):96,
    (13,'y',  False ,True ):97,
    (-1,'y',  False ,True ):98,
    (1,'y',  True ,True ):99,
    (2,'y',  True ,True ):100,
    (3,'y',  True ,True ):101,
    (4,'y',  True ,True ):102,
    (5,'y',  True ,True ):103,
    (6,'y',  True ,True ):104,
    (7,'y',  True ,True ):105,
    (8,'y',  True ,True ):106,
    (9,'y',  True ,True ):107,
    (10,'y',  True ,True ):108,
    (11,'y',  True ,True ):109,
    (12,'y',  True ,True ):110,
    (13,'y',  True ,True ):111,
    (-1,'y',  True ,True ):112
}


A1_OBS = ['a', 'x']
A2_OBS = ['s']
A3_OBS = ['c', 'y']

ACTIONS = {
    0:[0,0],
    1:[0,1],
    2:[1,0],
    3:[1,1],
}

ACTIONS_INV ={
    (0,0):0,
    (0,1):1,
    (1,0):2,
    (1,1):3,
}



def epsilon_decay(min_epsilon, episode, max_epochs):
    if episode <= 0.6*max_epochs:
        return 1.0
    
    initial_epsilon = 1.0
    return max(min_epsilon, initial_epsilon-(episode/(max_epochs)))
    # return min_epsilon

def get_action(q_table, agent_j_in_dead_state, agent_k_in_dead_state, row_num, epsilon):
    # Both agents are in dead state, only action [0,0] is possible
    if agent_j_in_dead_state and agent_k_in_dead_state:
        return 0  
    
    # If one agent is in dead state, limit actions for that agent 
    elif agent_j_in_dead_state:
        if random.uniform(0, 1) < epsilon:
            return np.argmin(q_table[row_num][[0,1]])  # Explore
        else:
            return np.argmax(q_table[row_num][[0,1]])  # Exploit
    elif agent_k_in_dead_state:
        if random.uniform(0, 1) < epsilon:
            return  2*np.argmin(q_table[row_num][[0,2]])  # Explore
        else:
            return  2*np.argmax(q_table[row_num][[0,2]])  # Exploit

    # Neither agent is in dead state, all actions possible
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Explore
    else:
        return np.argmax(q_table[row_num])  # Exploit


def q_training(env, epochs=10000, alpha = 0.1, gamma=0.1, min_epsilon=0.1, print_process=False):
    q_1 = np.zeros((len(S_1), env.action_space.n))
    # q_2 = np.zeros((len(S_2), env.action_space.n))
    q_3 = np.zeros((len(S_3), env.action_space.n))
    
    for episode in range(epochs):
        if (print_process and episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")
        
        config, info = env.reset()
        
        curr_event=info['curr_event']
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = config
        
        s_1, s_3 = -1, -1
        
        terminated = False
        
        reward_1, reward_3 = 0, 0
        
        agent_1_in_dead_state, agent_2_in_dead_state, agent_3_in_dead_state = False, False, False
        
        a1_action, a3_action = None, None

        epsilon = epsilon_decay(min_epsilon, episode, epochs)
        # if (episode%10000==0):
        #     print(f"Episode: {episode}, Epsilon: {epsilon}")
        
        while not terminated:
            if curr_event in A1_OBS:
                agent_id = 1
                
                next_s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                if s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[s_1][a1_action] += alpha * (reward_1 + gamma * np.max(q_1[next_s_1]) - q_1[s_1][a1_action])

                    reward_1 = 0
                                
                a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=next_s_1, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a1_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_1 += comm_cost
                
                s_1 = next_s_1
                
            if curr_event in A3_OBS:
                agent_id = 3
               
                next_s_3 = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                if s_3 != -1 :
                    # Q-value update for agent 3
                    q_3[s_3][a3_action] += alpha * (reward_3 + gamma * np.max(q_3[next_s_3]) - q_3[s_3][a3_action])
                    reward_3 = 0
                
                a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=next_s_3, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a3_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_3 += comm_cost
                
                s_3 = next_s_3
         
            agent_1_in_dead_state = agent_1_belief == -1
            
            agent_2_in_dead_state = agent_2_belief == -1
            
            agent_3_in_dead_state = agent_3_belief == -1
            
            curr_event=info['curr_event']
    
        
        # Q-value update for agents who took action
        reward_1 += penalty
        q_1[s_1][a1_action] += alpha * (reward_1 + gamma * 0 - q_1[s_1][a1_action])

        # reward_2 += penalty
        # q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * 0 - q_2[prev_s_2][a2_action])
        reward_3 += penalty
        q_3[s_3][a3_action] += alpha * (reward_3 + gamma * 0 - q_3[s_3][a3_action])
        
        
    # print(a1_action_count)
    # print(a3_action_count)
    # print(action_dict)
    return q_1, q_3






