import numpy as np
import gymnasium as gym
import pandas as pd
import random
import sys
sys.path.insert(0, './problem_w_unobservable_events')
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = 'three_agents_benchmark'

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

def get_action(q_table, agent_j_in_dead_state, agent_k_in_dead_state, row_num, epsilon):
    
    # Both agents are in dead state, only action [0,0] is possible
    if agent_j_in_dead_state and agent_k_in_dead_state:
        return 0  
    
    # If one agent is in dead state, limit actions for that agent
    elif agent_j_in_dead_state or agent_k_in_dead_state: 
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 1) if agent_j_in_dead_state else random.choice([0,2])  # Explore
        else:
            if agent_j_in_dead_state:
                return np.argmax(q_table[row_num][[0,1]])  # Exploit
            else:
                return np.argmax(q_table[row_num][[0,2]])  # Exploit
    
    # Neither agent is in dead state, all actions possible
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Explore
    else:
        return  np.argmax(q_table[row_num])  # Exploit

def q_training(env, epochs=10000, alpha = 0.1, gamma=0.1, epsilon=0.1, print_process=False):
    q_1 = np.zeros((len(S), env.action_space.n))
    q_2 = np.zeros((len(S), env.action_space.n))
    q_3 = np.zeros((len(S), env.action_space.n))
    for episode in range(epochs):
        if (print_process and episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")
            
        config, info = env.reset()
        
        curr_event=info['curr_event']
        word = info['string']
        print(word)
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = config
        
        prev_s_1 = -1
        prev_s_2 = -1
        prev_s_3 = -1
        
        terminated = False
        
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0
        
        agent_1_in_dead_state = False
        agent_2_in_dead_state = False
        agent_3_in_dead_state = False
        
        a1_action = None
        a2_action = None
        a3_action = None
        
        while not terminated:
            if curr_event == 'a':
                agent_id = 1
                
                s_1 = S[(agent_1_belief, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                # if prev_s_1 != -1 :
                #     # Q-value update for agent 1
                #     q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * np.max(q_1[s_1]) - q_1[prev_s_1][a1_action])
                #     reward_1 = 0
                
                a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_1, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a1_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                
                agent_3_in_dead_state = agent_3_belief == -1
                
                reward_1 += comm_cost
                
                curr_event=info['curr_event']
                
                prev_s_1 = s_1
                
            if curr_event == 'b':
                agent_id = 2
                
                s_2 = S[(agent_2_belief, agent_1_in_dead_state, agent_3_in_dead_state)]
                
                # if prev_s_2 != -1 :
                #     # Q-value update for agent 2
                #     q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * np.max(q_2[s_2]) - q_2[prev_s_2][a2_action])
                #     reward_2 = 0
                
                a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_2, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a2_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                agent_3_in_dead_state = agent_3_belief == -1
                
                reward_2 += comm_cost
                
                curr_event=info['curr_event']
                
                prev_s_2 = s_2
                
            if curr_event == 'c':
                agent_id = 3
                
                s_3 = S[(agent_3_belief, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                # if prev_s_3 != -1 :
                #     # Q-value update for agent 3
                #     q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * np.max(q_3[s_3]) - q_3[prev_s_3][a3_action])
                #     reward_3 = 0
                
                a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=s_3, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a3_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                agent_2_in_dead_state = agent_2_belief == -1
                
                reward_3 += comm_cost
                
                curr_event=info['curr_event']
                
                prev_s_3 = s_3
    
    
        # Final Q-value update at the end of the episode
        if a1_action is not None:
            reward_1 += penalty
            q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * 0 - q_1[prev_s_1][a1_action])
    
        if a2_action is not None:
            reward_2 += penalty
            q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * 0 - q_2[prev_s_2][a2_action])
    
        if a3_action is not None:
            reward_3 += penalty
            q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * 0 - q_3[prev_s_3][a3_action])
    
    return q_1, q_2, q_3

env = gym.make('ThreeAgentsEnv-v0', render_mode="human", string_mode="training")
q_1, q_2, q_3 = q_training(env, epochs=1, alpha = 0.01, gamma=0.5, epsilon=0.1, print_process=False)

q_1_df = pd.DataFrame(q_1, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])    
q_2_df = pd.DataFrame(q_2, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])    
q_3_df = pd.DataFrame(q_3, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])  
  
q_1_df.to_csv(f"{FOLDER_NAME}/three_agents_q1.csv", index=False)
q_2_df.to_csv(f"{FOLDER_NAME}/three_agents_q2.csv", index=False)
q_3_df.to_csv(f"{FOLDER_NAME}/three_agents_q3.csv", index=False)