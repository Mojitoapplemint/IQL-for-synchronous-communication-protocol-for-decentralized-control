import gymnasium as gym
import numpy as np
import pandas as pd
import benchmark_env
from benchmark_q import q_main

env = gym.make('BenchmarkEnv-v1', render_mode=None)

num_sessions=1000

agent_1_com_count = {
    1:[0,0],
    2:[0,0],
    3:[0,0],
    4:[0,0],
    5:[0,0],
    6:[0,0],
    7:[0,0],
    -1:[0,0],    
}

agent_2_com_count = {
    1:[0,0],
    2:[0,0],
    3:[0,0],
    4:[0,0],
    5:[0,0],
    6:[0,0],
    7:[0,0],
    -1:[0,0],    
}

for session in range(num_sessions):
    print(str(100*session/num_sessions)+"%","done" , end="\r")
    
    q_1, q_2 = q_main(env, epochs=2000, alpha=0.01)
    
    env = gym.make('BenchmarkEnv-v1', render_mode=None)
        
    for i in range(2):
        terminated = False

        v_state, info = env.reset()
        
        curr_event = info["curr_event"]
        
        _, agent_1_belief, agent_2_belief = v_state

        for _ in range(2):
            if curr_event == 'a':
                agent_id = 1
                a1_row_num = agent_1_belief
                
                a1_action = q_1[a1_row_num]
                
                a1_action = np.argmax(a1_action)
                
                agent_1_com_count[agent_1_belief][a1_action] += 1
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
                
                system_state, agent_1_belief, agent_2_belief= v_state
                
                curr_event = info["curr_event"]
                
            if curr_event == 'b':
                agent_id = 2
                a2_row_num = agent_2_belief
                
                a2_action = q_2[a2_row_num]
    
                a2_action = np.argmax(a2_action)
                
                agent_2_com_count[agent_2_belief][a2_action] += 1
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
                
                system_state, agent_1_belief, agent_2_belief = v_state
                
                curr_event = info["curr_event"]

print("Agent 1 Communication Protocol and count:")
for belief, count in agent_1_com_count.items():
    if count != [0,0]:
        print(f"Agent 1 belief: {belief}, Do Not Communicate: {count[0]}, Communicate: {count[1]}")
    
    
print("\nAgent 2 Communication Protocol and count:")
for belief, count in agent_2_com_count.items():
    if count != [0,0]:
        print(f"Agent 2 belief: {belief}, Do Not Communicate: {count[0]}, Communicate: {count[1]}")