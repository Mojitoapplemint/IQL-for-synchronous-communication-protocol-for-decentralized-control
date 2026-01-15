import gymnasium as gym
import numpy as np
import pandas as pd
import benchmark_env
from benchmark_q import q_main

env = gym.make('BenchmarkEnv-v0', render_mode=None)

num_sessions=1000

agent_1_communication_protocol=[[0,0], [0,0]]
agent_2_communication_protocol=[[0,0], [0,0]]


for session in range(num_sessions):
    print(str(100*session/num_sessions)+"%","done" , end="\r")
    
    q_1, q_2 = q_main(env, epochs=2000, alpha=0.01)
    
    if q_1[0,0]>q_1[0,1]:
        agent_1_communication_protocol[0][0]+=1
    else:
        agent_1_communication_protocol[0][1]+=1
    if q_1[1,0]>q_1[1,1]:
        agent_1_communication_protocol[1][0]+=1
    else:
        agent_1_communication_protocol[1][1]+=1
        
    if q_2[0,0]>q_2[0,1]:
        agent_2_communication_protocol[0][0]+=1
    else:
        agent_2_communication_protocol[0][1]+=1
    if q_2[1,0]>q_2[1,1]:
        agent_2_communication_protocol[1][0]+=1
    else:
        agent_2_communication_protocol[1][1]+=1

agent_1_communication_protocol_df = pd.DataFrame(agent_1_communication_protocol, columns=['Do Not Communicate', 'Communicate'], index=['observe "a" in belief state 1', 'observe "a" in belief state 3'])
agent_2_communication_protocol_df = pd.DataFrame(agent_1_communication_protocol, columns=['Do Not Communicate', 'Communicate'], index=['observe "b" in belief state 1', 'observe "b" in belief state 2'])
agent_1_communication_protocol_df.to_csv(f"./benchmark_problem/agent_1_communication_protocol.csv")
agent_2_communication_protocol_df.to_csv(f"./benchmark_problem/agent_2_communication_protocol.csv")