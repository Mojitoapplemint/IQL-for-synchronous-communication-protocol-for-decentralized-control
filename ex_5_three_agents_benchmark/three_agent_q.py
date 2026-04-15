import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = 'ex_5_three_agents_benchmark'


ACTIONS = {
    0:[0,0],
    1:[0,1],
    2:[1,0],
    3:[1,1],
}

def epsilon_decay(min_epsilon, episode, max_epochs):
    # return 0.1
    
    if episode <= 0.2*max_epochs:
        return 1.0
    
    initial_epsilon = 1.0
    return max(min_epsilon, initial_epsilon-(episode/(max_epochs)))

def get_action(q_table, row_num, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Explore: random action
    else:
        return  np.argmax(q_table[row_num])  # Exploit: best action from Q-table

def q_training(env, max_epochs=10000, alpha = 0.1, gamma=0.1, min_epsilon=0.1, print_process=False):
    q_1 = np.zeros((23, env.action_space.n))
    q_2 = np.zeros((23, env.action_space.n))
    q_3 = np.zeros((23, env.action_space.n))
    for episode in range(max_epochs):
        if (print_process and episode%100==0):
            print(str(100*episode/max_epochs)+"%","done" , end="\r")
            
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
        
        a1_action = None
        a2_action = None
        a3_action = None
        
        epsilon = epsilon_decay(min_epsilon, episode, max_epochs)
        
        while not terminated:
            if curr_event == 'a':
                agent_id = 1
                
                s_1 = agent_1_belief
                
                if prev_s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * np.max(q_1[s_1]) - q_1[prev_s_1][a1_action])
                    reward_1 = 0
                
                a1_action = get_action(q_1, row_num=s_1, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a1_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_1 += comm_cost
                                
                prev_s_1 = s_1
                
            if curr_event == 'b':
                agent_id = 2
                
                s_2 = agent_2_belief
                
                if prev_s_2 != -1 :
                    # Q-value update for agent 2
                    q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * np.max(q_2[s_2]) - q_2[prev_s_2][a2_action])
                    reward_2 = 0
                
                a2_action = get_action(q_2, row_num=s_2, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a2_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config                
                
                reward_2 += comm_cost
                                
                prev_s_2 = s_2
                
            if curr_event == 'c':
                agent_id = 3
                
                s_3 = agent_3_belief
                
                if prev_s_3 != -1 :
                    # Q-value update for agent 3
                    q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * np.max(q_3[s_3]) - q_3[prev_s_3][a3_action])
                    reward_3 = 0
                
                a3_action = get_action(q_3, row_num=s_3, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a3_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_3 += comm_cost
                
                prev_s_3 = s_3
            
            curr_event=info['curr_event']

        if a1_action is not None:
            reward_1 += penalty
            q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * 0 - q_1[prev_s_1][a1_action])

        if a2_action is not None:
            reward_2 += penalty
            q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * 0 - q_2[prev_s_2][a2_action])

        if a3_action is not None:
            reward_3 += penalty
            q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * 0 - q_3[prev_s_3][a3_action])
        
        # print()
        # print(a1_action, a2_action, a3_action)
        # print(reward_1, reward_2, reward_3)
    
    return q_1, q_2, q_3

