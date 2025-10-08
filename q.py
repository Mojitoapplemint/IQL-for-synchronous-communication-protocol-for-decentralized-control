import gymnasium as gym
import numpy as np
import pandas as pd
import env
import warnings
warnings.filterwarnings("ignore")

LOCAL_STATES_1 = {
    (1,'a'):0,
    (3,'a'):1,
    (2,'s'):2,
    (4,'s'):3,
    (5,'s'):4,
    (6,'$'):5,
    (7,'$'):6,
    (-1,'$'):7,
}


LOCAL_STATES_2 = {
    (1,'b'):0,
    (2,'b'):1,
    (3,'s'):2,
    (4,'s'):3,
    (5,'s'):4,
    (6,'$'):5,
    (7,'$'):6,
    (-1,'$'):7,
}

def get_action(epsilon, q, observation):
    if np.random.rand() < epsilon:
        # print("Exploration")
        return np.random.randint(0, 2)  # Explore: random action
    else:
        # print("Exploitation")
        return np.argmax(q[observation])  # Exploit: best action from Q-table

def q_main(env, model_name, epochs=10000, epsilon=0.1, gamma=0.1, alpha=0.1):
    q_1 =np.zeros(shape=(13, 2))  # 13 local states, 2 actions
    q_2 =np.zeros(shape=(13, 2))  # 13 local states, 2 actions
    
    for epoch in range(epochs):
        if (epoch%100==0):
            print(str(100*epoch/epochs)+"%","done" , end="\r")
        
        terminated = False
        truncated = False
        

        config, info = env.reset()
        _ , agent_1_observation, agent_2_observation = config
        
        
        while not (terminated or truncated):
            curr_alphabet=info["input_alphabet"]    
            if curr_alphabet == 'a':
                agent_id = 1
                state_num_1 = LOCAL_STATES_1[(agent_1_observation, curr_alphabet)]
                
                communicate = get_action(epsilon, q_1, agent_1_observation)
                
            elif curr_alphabet == 'b':
                agent_id = 2
                state_num_2 = LOCAL_STATES_2[(agent_2_observation, curr_alphabet)] 
                
                communicate = get_action(epsilon, q_2, agent_2_observation)
            else:
                raise ValueError(f"Invalid alphabet, curr_alphabet: {curr_alphabet}")
            
            action = (agent_id, communicate)
            
            config, reward, terminated, truncated, info = env.step(action)
            _ , agent_1_observation, agent_2_observation = config
            
            next_curr_alphabet=info["input_alphabet"]
            
            if next_curr_alphabet == '$':
                next_state_num_1 = LOCAL_STATES_1[(agent_1_observation, next_curr_alphabet)]
                next_state_num_2 = LOCAL_STATES_2[(agent_2_observation, next_curr_alphabet)]
                
                # Q-learning update for both agents
                q_1[state_num_1, communicate] += alpha * (reward + gamma * np.max(q_1[next_state_num_1]) - q_1[state_num_1, communicate])
                q_2[state_num_2, communicate] += alpha * (reward + gamma * np.max(q_2[next_state_num_2]) - q_2[state_num_2, communicate])
                
                if not (terminated or truncated):
                    raise ValueError("Should be terminated or truncated at '$'")
            else:
                if agent_id == 1:
                    next_state_num_1 = LOCAL_STATES_1[(agent_1_observation, curr_alphabet)]
                    
                    # Q-learning update
                    q_1[state_num_1, communicate] += alpha * (reward + gamma * np.max(q_1[next_state_num_1]) - q_1[state_num_1, communicate])
                    
                    state_num_1 = next_state_num_1
                elif agent_id == 2:
                    next_state_num_2 = LOCAL_STATES_2[(agent_2_observation, curr_alphabet)]
                    
                    # Q-learning update
                    q_2[state_num_2, communicate] += alpha * (reward + gamma * np.max(q_2[next_state_num_2]) - q_2[state_num_2, communicate])
                    
                    state_num_2 = next_state_num_2
    
    q_1_df = pd.DataFrame(q_1, columns=['Do Not Communicate', 'Communicate'], index=[i for i in range(13)])
    q_2_df = pd.DataFrame(q_2, columns=['Do Not Communicate', 'Communicate'], index=[i for i in range(13)])
    q_1_df.to_csv(f"./{model_name}_q_1.csv")
    q_2_df.to_csv(f"./{model_name}_q_2.csv")
    
env = gym.make('ComposedEnv-v0', render_mode='human')
q_main(env, 'demo', epochs=10)