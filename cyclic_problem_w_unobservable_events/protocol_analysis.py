import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from word_generator import WordGenerator
import gymnasium as gym
import cyclic_problem_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


m_bottom={
    1:"{1}",
    2:"{2}",
    3:"{1,3}",
    4:"{4}",
    5:"{5}",
    6:"{6,1}",
    7:"{7}",
    -1:"{-1}",
}

PHI = {
    (False, 1 ):0,
    (False, 2 ):1,
    (False, 3 ):2,
    (False, 4 ):3,
    (False, 5 ):4,
    (False, 6 ):5,
    (False, 7 ):6,
    (False,-1 ):7,
    (True, 1 ):8,
    (True, 2 ):9,
    (True, 3 ):10,
    (True, 4 ):11,
    (True, 5 ):12,
    (True, 6 ):13,
    (True, 7 ):14,
    (True,-1 ):15,
}


protocols_data = pd.read_csv("cyclic_problem_w_unobservable_events/protocols_with_stats.csv")

returns_data = protocols_data[['Agent 1 Average Cumulative Reward', 'Agent 2 Average Cumulative Reward']]


a = protocols_data[(protocols_data['Agent 1 Average Cumulative Reward']==-40.48) & (protocols_data['Agent 2 Average Cumulative Reward']==0)]
# a = protocols_data[(protocols_data['Agent 1 Average Cumulative Reward']==0) & (protocols_data['Agent 2 Average Cumulative Reward']==-40.48)]


a1_protocol_list=[]
a2_protocol_list=[]



for index, row in a.iterrows():
    # print(f"{index} / {len(successful_protocols)}")

    
    protocol = row["Communication Protocols"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    # print(protocol)
    
    q_1 = protocol[:16]
    q_2 = protocol[16:]
    
    
    env = gym.make("CyclicEnv-v0", render_mode = None, string_mode="stats")
    
    a1_communication_protocol = {
           ( 1, False):[0,0],
           ( 2, False):[0,0],
           ( 3, False):[0,0],
           ( 4, False):[0,0],
           ( 5, False):[0,0],
           ( 6, False):[0,0],
           ( 7, False):[0,0],
           (-1, False):[0,0],
           ( 1, True):[0,0],
           ( 2, True):[0,0],
           ( 3, True):[0,0],
           ( 4, True):[0,0],
           ( 5, True):[0,0],
           ( 7, True):[0,0],
           ( 6, True):[0,0],
           (-1, True):[0,0],
        }
    
    a2_communication_protocol = {
            ( 1, False):[0,0],
            ( 2, False):[0,0],
            ( 3, False):[0,0],
            ( 4, False):[0,0],
            ( 5, False):[0,0],
            ( 6, False):[0,0],
            ( 7, False):[0,0],
            (-1, False):[0,0],
            ( 1, True):[0,0],
            ( 2, True):[0,0],
            ( 3, True):[0,0],
            ( 4, True):[0,0],
            ( 5, True):[0,0],
            ( 6, True):[0,0],
            ( 7, True):[0,0],
            (-1, True):[0,0],
        }
    
    
    communicate_count = [0,0,0,0]
    
    for i in range (1000):
        terminated = False
        simulation_result = False


        config, info = env.reset()

        global_state, agent_1_belief, agent_2_belief = config

        curr_symbol=info['input_alphabet']

        agent_1_prev_row_num = -1
        agent_2_prev_row_num = -1

        agent_1_in_dead_state = False
        agent_2_in_dead_state = False

        reward_1=0
        reward_2=0
        
        t_1=1
        t_2=1        
        
        while not (terminated):
            if curr_symbol=='a':
                
                agent_id=1
            
                if agent_2_in_dead_state:
                    agent_1_communicate = 0
                else:
                    agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
                    agent_1_communicate = q_1[agent_1_row_num]
                    
                if agent_1_communicate ==1:
                    communicate_count[0] += 1
                    (a1_communication_protocol[agent_1_belief, agent_2_in_dead_state])[0] += 1
                else:
                    communicate_count[1] += 1
                    (a1_communication_protocol[agent_1_belief, agent_2_in_dead_state])[1] += 1


                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                system_state, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                
                comm_cost, penalty = reward
                
                reward_1 += comm_cost
                
                curr_symbol=info['input_alphabet']
                
                agent_1_prev_row_num = agent_1_row_num
                
                
                       
            if curr_symbol=='b':

                agent_id=2
                
                if agent_1_in_dead_state:
                    agent_2_communicate = 0
                else:
                    agent_2_row_num = PHI[(agent_1_in_dead_state, agent_2_belief)]
                    agent_2_communicate = q_2[agent_2_row_num]
                
                
                if agent_2_communicate ==1:
                    
                    communicate_count[2] += 1
                    (a2_communication_protocol[agent_2_belief, agent_1_in_dead_state])[0] += 1  
                else:
                    communicate_count[3] += 1
                    (a2_communication_protocol[agent_2_belief, agent_1_in_dead_state])[1] += 1  
                
                
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                
                system_state, agent_1_belief, agent_2_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                comm_cost, penalty = reward
                
                reward_2 += comm_cost
                                
                curr_symbol=info['input_alphabet']
                
                agent_2_prev_row_num = agent_2_row_num
        
    a1_protocol_list.append(a1_communication_protocol)
    a2_protocol_list.append(a2_communication_protocol)
        
for i in range(len(a1_protocol_list)):
    a1_protocol = a1_protocol_list[i]
    a2_protocol = a2_protocol_list[i]
    
    
    print(f"\n================== {i}'th protocol ==================\nAgent 1 Communication Protocol:")
    for belief_state in a1_protocol:
        # print(belief_state)
        if (a1_protocol[belief_state] != [0,0] and not belief_state[1]):
            print("In state ("+ m_bottom[belief_state[0]]+ ", "+str(belief_state[1])+ ") Num Communicate: " + str(a1_protocol[belief_state][0]) + " Num Not Communicate: " + str(a1_protocol[belief_state][1]))

    print(f"\nAgent 2 Communication Protocol:")
    for belief_state in a2_protocol:
        if (a2_protocol[belief_state] != [0,0] and not belief_state[1]):
            print("In state ("+ m_bottom[belief_state[0]]+ ", "+str(belief_state[1])+ ") Num Communicate: " + str(a2_protocol[belief_state][0]) + " Num Not Communicate: " + str(a2_protocol[belief_state][1]))

