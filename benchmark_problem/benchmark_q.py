import sys
sys.path.insert(0, './benchmark_problem')
import gymnasium as gym
import numpy as np
import pandas as pd
import benchmark_env
import warnings
warnings.filterwarnings("ignore")

LOCAL_STATES_1 = {
    (1,'a'):0,
    (3,'a'):1,
    (6,'$'):2,
    (7,'$'):3,
    (-1,'$'):4,
}


LOCAL_STATES_2 = {
    (1,'b'):0,
    (2,'b'):1,
    (6,'$'):2,
    (7,'$'):3,
    (-1,'$'):4,
}

def get_action(epsilon, q, observation):
    if np.random.rand() < epsilon:
        # print("Exploration")
        return np.random.randint(0, 2)  # Explore: random action
    else:
        # print("Exploitation")
        return np.argmax(q[observation])  # Exploit: best action from Q-table

def q_main(env, epochs=10000, epsilon=0.1, gamma=0.1, alpha=0.1, print_process=False):
    q_1 =np.zeros(shape=(5, 2))
    q_2 =np.zeros(shape=(5, 2))
    
    for epoch in range(epochs):
        if (print_process and epoch%100==0):
            print(str(100*epoch/epochs)+"%","done" , end="\r")
        
        config, info = env.reset()
        _ , agent_1_belief, agent_2_belief = config
        
        curr_symbol=info['input_alphabet']
        
        if curr_symbol == 'a':
            agent_id = 1
            state_num_1 = LOCAL_STATES_1[(agent_1_belief, curr_symbol)]
            
            communicate_1 = get_action(epsilon, q_1, state_num_1)
            
            action = (agent_id, communicate_1)
            
            config, reward, _, _, info = env.step(action)
            _ , agent_1_belief, agent_2_belief = config
            
            curr_symbol=info["input_alphabet"]
            
            state_num_2 = LOCAL_STATES_2[(agent_2_belief, curr_symbol)]
            
            communicate_2 = get_action(epsilon, q_2, state_num_2)
            
            agent_id = 2
            
            action = (agent_id, communicate_2)
            
            
        
        elif curr_symbol == 'b':
            agent_id = 2
            state_num_2 = LOCAL_STATES_2[(agent_2_belief, curr_symbol)] 
            
            communicate_2 = get_action(epsilon, q_2, state_num_2)
            
            action = (agent_id, communicate_2)
            
            config, reward, _, _, info = env.step(action)
            _ , agent_1_belief, agent_2_belief = config
            
            curr_symbol=info["input_alphabet"]
            
            state_num_1 = LOCAL_STATES_1[(agent_1_belief, curr_symbol)]
            
            communicate_1 = get_action(epsilon, q_1, state_num_1)
            
            agent_id = 1
            
            action = (agent_id, communicate_1)
            
        config, reward, _, _, info = env.step(action)
        _ , agent_1_belief, agent_2_belief = config
        
        next_curr_alphabet=info["input_alphabet"]
        
        next_state_num_1 = LOCAL_STATES_1[(agent_1_belief, next_curr_alphabet)]
        next_state_num_2 = LOCAL_STATES_2[(agent_2_belief, next_curr_alphabet)]
        
        # Q-learning update for both agents
        q_1[state_num_1, communicate_1] += alpha * (reward + gamma * np.max(q_1[next_state_num_1]) - q_1[state_num_1, communicate_1])
        q_2[state_num_2, communicate_2] += alpha * (reward + gamma * np.max(q_2[next_state_num_2]) - q_2[state_num_2, communicate_2])
            
    
    return q_1, q_2
    
    
env = gym.make('BenchmarkEnv-v0', render_mode=None)
q_1, q_2 = q_main(env, epochs=100000, alpha=0.01)

q_1_df = pd.DataFrame(q_1, columns=['Do Not Communicate', 'Communicate'], index=[i for i in range(5)])
q_2_df = pd.DataFrame(q_2, columns=['Do Not Communicate', 'Communicate'], index=[i for i in range(5)])
q_1_df.to_csv(f"./benchmark_problem/demo_q_1.csv")
q_2_df.to_csv(f"./benchmark_problem/demo_q_2.csv")