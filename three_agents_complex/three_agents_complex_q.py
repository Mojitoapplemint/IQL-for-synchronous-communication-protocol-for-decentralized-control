import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_complex_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = 'three_agents_complex'

S_1={
    (1 ,'a', False, False):0,
    (2 ,'a', False, False):1,
    (3 ,'a', False, False):2,
    (4 ,'a', False, False):3,
    (5 ,'a', False, False):4,
    (6 ,'a', False, False):5,
    (7 ,'a', False, False):6,
    (8 ,'a', False, False):7,
    (9 ,'a', False, False):8,
    (-1,'a', False, False):9,
    (1 ,'a', True, False):10,
    (2 ,'a', True, False):11,
    (3 ,'a', True, False):12,
    (4 ,'a', True, False):13,
    (5 ,'a', True, False):14,
    (6 ,'a', True, False):15,
    (7 ,'a', True, False):16,
    (8 ,'a', True, False):17,
    (9 ,'a', True, False):18,
    (-1,'a', True, False):19,
    (1 ,'a', False, True):20,
    (2 ,'a', False, True):21,
    (3 ,'a', False, True):22,
    (4 ,'a', False, True):23,
    (5 ,'a', False, True):24,
    (6 ,'a', False, True):25,
    (7 ,'a', False, True):26,
    (8 ,'a', False, True):27,
    (9 ,'a', False, True):28,
    (-1,'a', False, True):29,
    (1 ,'a', True, True):30,
    (2 ,'a', True, True):31,
    (3 ,'a', True, True):32,
    (4 ,'a', True, True):33,
    (5 ,'a', True, True):34,
    (6 ,'a', True, True):35,
    (7 ,'a', True, True):36,
    (8 ,'a', True, True):37,
    (9 ,'a', True, True):38,
    (-1,'a', True, True):39,
    (1 ,'x', False, False):40,
    (2 ,'x', False, False):41,
    (3 ,'x', False, False):42,
    (4 ,'x', False, False):43,
    (5 ,'x', False, False):44,
    (6 ,'x', False, False):45,
    (7 ,'x', False, False):46,
    (8 ,'x', False, False):47,
    (9 ,'x', False, False):48,
    (-1,'x', False, False):49,
    (1 ,'x', True, False):50,
    (2 ,'x', True, False):51,
    (3 ,'x', True, False):52,
    (4 ,'x', True, False):53,
    (5 ,'x', True, False):54,
    (6 ,'x', True, False):55,
    (7 ,'x', True, False):56,
    (8 ,'x', True, False):57,
    (9 ,'x', True, False):58,
    (-1,'x', True, False):59,
    (1 ,'x', False, True):60,
    (2 ,'x', False, True):61,
    (3 ,'x', False, True):62,
    (4 ,'x', False, True):63,
    (5 ,'x', False, True):64,
    (6 ,'x', False, True):65,
    (7 ,'x', False, True):66,
    (8 ,'x', False, True):67,
    (9 ,'x', False, True):68,
    (-1,'x', False, True):69,
    (1 ,'x', True, True):70,
    (2 ,'x', True, True):71,
    (3 ,'x', True, True):72,
    (4 ,'x', True, True):73,
    (5 ,'x', True, True):74,
    (6 ,'x', True, True):75,
    (7 ,'x', True, True):76,
    (8 ,'x', True, True):77,
    (9 ,'x', True, True):78,
    (-1,'x', True, True):79,
}

S_2={
    (1 ,'b', False, False):0,
    (2 ,'b', False, False):1,
    (3 ,'b', False, False):2,
    (4 ,'b', False, False):3,
    (5 ,'b', False, False):4,
    (6 ,'b', False, False):5,
    (7 ,'b', False, False):6,
    (8 ,'b', False, False):7,
    (9 ,'b', False, False):8,
    (-1,'b', False, False):9,
    (1 ,'b', True, False):10,
    (2 ,'b', True, False):11,
    (3 ,'b', True, False):12,
    (4 ,'b', True, False):13,
    (5 ,'b', True, False):14,
    (6 ,'b', True, False):15,
    (7 ,'b', True, False):16,
    (8 ,'b', True, False):17,
    (9 ,'b', True, False):18,
    (-1,'b', True, False):19,
    (1 ,'b', False, True):20,
    (2 ,'b', False, True):21,
    (3 ,'b', False, True):22,
    (4 ,'b', False, True):23,
    (5 ,'b', False, True):24,
    (6 ,'b', False, True):25,
    (7 ,'b', False, True):26,
    (8 ,'b', False, True):27,
    (9 ,'b', False, True):28,
    (-1,'b', False, True):29,
    (1 ,'b', True, True):30,
    (2 ,'b', True, True):31,
    (3 ,'b', True, True):32,
    (4 ,'b', True, True):33,
    (5 ,'b', True, True):34,
    (6 ,'b', True, True):35,
    (7 ,'b', True, True):36,
    (8 ,'b', True, True):37,
    (9 ,'b', True, True):38,
    (-1,'b', True, True):39,
}

S_3={
    (1 ,'c', False, False):0,
    (2 ,'c', False, False):1,
    (3 ,'c', False, False):2,
    (4 ,'c', False, False):3,
    (5 ,'c', False, False):4,
    (6 ,'c', False, False):5,
    (7 ,'c', False, False):6,
    (8 ,'c', False, False):7,
    (9 ,'c', False, False):8,
    (-1,'c', False, False):9,
    (1 ,'c', True, False):10,
    (2 ,'c', True, False):11,
    (3 ,'c', True, False):12,
    (4 ,'c', True, False):13,
    (5 ,'c', True, False):14,
    (6 ,'c', True, False):15,
    (7 ,'c', True, False):16,
    (8 ,'c', True, False):17,
    (9 ,'c', True, False):18,
    (-1,'c', True, False):19,
    (1 ,'c', False, True):20,
    (2 ,'c', False, True):21,
    (3 ,'c', False, True):22,
    (4 ,'c', False, True):23,
    (5 ,'c', False, True):24,
    (6 ,'c', False, True):25,
    (7 ,'c', False, True):26,
    (8 ,'c', False, True):27,
    (9 ,'c', False, True):28,
    (-1,'c', False, True):29,
    (1 ,'c', True, True):30,
    (2 ,'c', True, True):31,
    (3 ,'c', True, True):32,
    (4 ,'c', True, True):33,
    (5 ,'c', True, True):34,
    (6 ,'c', True, True):35,
    (7 ,'c', True, True):36,
    (8 ,'c', True, True):37,
    (9 ,'c', True, True):38,
    (-1,'c', True, True):39,
    (1 ,'x', False, False):40,
    (2 ,'x', False, False):41,
    (3 ,'x', False, False):42,
    (4 ,'x', False, False):43,
    (5 ,'x', False, False):44,
    (6 ,'x', False, False):45,
    (7 ,'x', False, False):46,
    (8 ,'x', False, False):47,
    (9 ,'x', False, False):48,
    (-1,'x', False, False):49,
    (1 ,'x', True, False):50,
    (2 ,'x', True, False):51,
    (3 ,'x', True, False):52,
    (4 ,'x', True, False):53,
    (5 ,'x', True, False):54,
    (6 ,'x', True, False):55,
    (7 ,'x', True, False):56,
    (8 ,'x', True, False):57,
    (9 ,'x', True, False):58,
    (-1,'x', True, False):59,
    (1 ,'x', False, True):60,
    (2 ,'x', False, True):61,
    (3 ,'x', False, True):62,
    (4 ,'x', False, True):63,
    (5 ,'x', False, True):64,
    (6 ,'x', False, True):65,
    (7 ,'x', False, True):66,
    (8 ,'x', False, True):67,
    (9 ,'x', False, True):68,
    (-1,'x', False, True):69,
    (1 ,'x', True, True):70,
    (2 ,'x', True, True):71,
    (3 ,'x', True, True):72,
    (4 ,'x', True, True):73,
    (5 ,'x', True, True):74,
    (6 ,'x', True, True):75,
    (7 ,'x', True, True):76,
    (8 ,'x', True, True):77,
    (9 ,'x', True, True):78,
    (-1,'x', True, True):79,
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
    q_1 = np.zeros((len(S_1), env.action_space.n))
    q_2 = np.zeros((len(S_2), env.action_space.n))
    q_3 = np.zeros((len(S_3), env.action_space.n))
    for episode in range(epochs):
        if (print_process and episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")
            
        config, info = env.reset()
        
        curr_event=info['curr_event']
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = config
        
        prev_s_1, prev_s_2, prev_s_3 = -1, -1, -1
        
        terminated = False
        
        reward_1, reward_2, reward_3 = 0, 0, 0
        
        agent_1_in_dead_state, agent_2_in_dead_state, agent_3_in_dead_state = False, False, False
        
        a1_action, a2_action, a3_action = None, None, None

        
        while not terminated:
            if curr_event == 'a':
                agent_id = 1
                
                s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                if prev_s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * np.max(q_1[s_1]) - q_1[prev_s_1][a1_action])
                    reward_1 = 0
                
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
                
                s_2 = S_2[(agent_2_belief, curr_event, agent_1_in_dead_state, agent_3_in_dead_state)]
                
                if prev_s_2 != -1 :
                    # Q-value update for agent 2
                    q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * np.max(q_2[s_2]) - q_2[prev_s_2][a2_action])
                    reward_2 = 0
                
                a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_2, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a2_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_2 += comm_cost
                
                prev_s_2 = s_2
                
            if curr_event == 'c':
                agent_id = 3
               
                s_3 = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                if prev_s_3 != -1 :
                    # Q-value update for agent 3
                    q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * np.max(q_3[s_3]) - q_3[prev_s_3][a3_action])
                    reward_3 = 0
                
                a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=s_3, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a3_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_3 += comm_cost
                
                prev_s_3 = s_3
            
            if curr_event == 'x':
                agent_id = 13
                
                s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
                s_3 = S_3[(agent_3_belief,curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                if prev_s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * np.max(q_1[s_1]) - q_1[prev_s_1][a1_action])
                    reward_1 = 0
                
                if prev_s_3 != -1 :
                    # Q-value update for agent 3
                    q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * np.max(q_3[s_3]) - q_3[prev_s_3][a3_action])
                    reward_3 = 0
                
                if agent_2_in_dead_state:
                    a1_action = 0
                    a3_action = 0
                else:
                    # Usual Epsilon-Greedy, assuming they are sending the communication to each other, so decide to communicate to agent 2 only or not
                    a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=True, row_num=s_1, epsilon=epsilon)
                    a3_action = get_action(q_3, agent_j_in_dead_state=True, agent_k_in_dead_state=agent_2_in_dead_state, row_num=s_3, epsilon=epsilon)
                    
                joint_action = (a1_action, a3_action)
                
                config, reward, terminated, _, info = env.step((agent_id, joint_action))
                
                comm_cost, penalty = reward
                
                comm_cost_1 , comm_cost_3 = comm_cost
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_1 += comm_cost_1
                reward_3 += comm_cost_3
                
                prev_s_3 = s_3
                prev_s_1 = s_1
            
            agent_1_in_dead_state = agent_1_belief == -1
            
            agent_2_in_dead_state = agent_2_belief == -1
            
            agent_3_in_dead_state = agent_3_belief == -1
            
            curr_event=info['curr_event']
    
        
        # Q-value update for agents who took action
        reward_1 += penalty
        q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * 0 - q_1[prev_s_1][a1_action])

        reward_2 += penalty
        q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * 0 - q_2[prev_s_2][a2_action])

        reward_3 += penalty
        q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * 0 - q_3[prev_s_3][a3_action])
        
        # print()
        # print(a1_action, a2_action, a3_action)
        # print(reward_1, reward_2, reward_3)
    
    return q_1, q_2, q_3

env = gym.make('ThreeAgentsComplexEnv-v0', render_mode='human', string_mode="training")

q_1, q_2, q_3 = q_training(env, epochs=10, alpha=0.001, gamma=0.1, epsilon=0.1, print_process=True)