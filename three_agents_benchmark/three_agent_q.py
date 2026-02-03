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
                
                reward_1 += np.sum(reward)
                
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
                
                reward_2 += np.sum(reward)
                
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
                
                reward_3 += np.sum(reward)
                
                curr_event=info['curr_event']
                
                prev_s_3 = s_3
        
        if agent_id == 1:
            reward_2 += penalty
            reward_3 += penalty
        elif agent_id == 2:
            reward_1 += penalty
            reward_3 += penalty
        elif agent_id == 3:
            reward_1 += penalty
            reward_2 += penalty
    
        # print(reward_1, reward_2, reward_3)
    
        # Final Q-value update at the end of the episode
        if a1_action is not None:
            q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * 0 - q_1[prev_s_1][a1_action])
    
        if a2_action is not None:
            q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * 0 - q_2[prev_s_2][a2_action])
    
        if a3_action is not None:
            q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * 0 - q_3[prev_s_3][a3_action])
    
    return q_1, q_2, q_3

env = gym.make('ThreeAgentsEnv-v0', render_mode=None, string_mode="training")
q_1, q_2, q_3 = q_training(env, epochs=10000, alpha = 0.001, gamma=0.5, epsilon=0.1, print_process=True)

q_1_df = pd.DataFrame(q_1, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])    
q_2_df = pd.DataFrame(q_2, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])    
q_3_df = pd.DataFrame(q_3, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])  
  
q_1_df.to_csv(f"{FOLDER_NAME}/three_agents_q1.csv", index=False)
q_2_df.to_csv(f"{FOLDER_NAME}/three_agents_q2.csv", index=False)
q_3_df.to_csv(f"{FOLDER_NAME}/three_agents_q3.csv", index=False)