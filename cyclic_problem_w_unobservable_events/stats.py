import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import cyclic_problem_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from cyclic_problem_q import q_training, S_1, S_2, FOLDER_NAME, A1_OBS, A2_OBS


GAMMA = 0.1

m_bottom={
    1:{1},
    2:{2},
    3:{1,3},
    4:{4},
    5:{5},
    6:{6,1},
    7:{7},
    -1:{-1},
}

successful_protocols = pd.read_csv(f"./{FOLDER_NAME}/successful_protocols.csv")
baseline_protocols = pd.read_csv(f"./{FOLDER_NAME}/baselines.csv")

protocols_df = pd.concat([baseline_protocols, successful_protocols], ignore_index=True)


# print(successful_protocols)

return_values_x = []
return_values_y = []
joint_return_values = []

communicate_counts = []

a1_protocol_list=[]
a2_protocol_list=[]

T_state_dict_list = []

for index, row in protocols_df.iterrows():
    # print(f"{index} / {len(successful_protocols)}")

    
    protocol = row["Communication Protocols"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    # print(protocol)
    
    q_1 = protocol[:len(S_1)]
    q_2 = protocol[len(S_1):]
    
    cumulative_reward = [0,0]
    
    env = gym.make("CyclicEnv-v1", render_mode = None, string_mode="simulation")
    
    a1_communication_protocol = {
           ( 1, 'a'):[0,0],
           ( 2, 'a'):[0,0],
           ( 3, 'a'):[0,0],
           ( 4, 'a'):[0,0],
           ( 5, 'a'):[0,0],
           ( 6, 'a'):[0,0],
           ( 7, 'a'):[0,0],
           (-1, 'a'):[0,0],
        }
    
    a2_communication_protocol = {
            ( 1, 'b'):[0,0],
            ( 2, 'b'):[0,0],
            ( 3, 'b'):[0,0],
            ( 4, 'b'):[0,0],
            ( 5, 'b'):[0,0],
            ( 6, 'b'):[0,0],
            ( 7, 'b'):[0,0],
            (-1, 'b'):[0,0],
        }
    
    T_state_dict={
        "(3, 3, 3)":0,
        "(7, 7, 7)":0,
        "(4, -1, -1)":0,
        "(7, -1, -1)":0,
        "(3, 3, -1)":0,
        "(7, 7, -1)":0,
        "(3, -1, 3)":0,
        "(7, -1, 7)":0,
        "(3, 3, 5)":0,
    }
    
    
    communicate_count = [0,0,0,0]
    
    for i in range (100):
        terminated = False
        simulation_result = False


        v_state, info = env.reset()

        global_state, agent_1_belief, agent_2_belief = v_state

        curr_event=info['curr_event']
        word = info['word']
        # print(word)

        s_1 = -1
        s_2 = -1

        reward_1=0
        reward_2=0
        
        t_1=1
        t_2=1        
        
        while not (terminated):
            if curr_event=='a':
                
                agent_id=1
                
                if s_1 != -1 :
                    # cumulative_reward[0] += (GAMMA**t_1)*reward_1
                    cumulative_reward[0] += reward_1
                    t_1+=1
                    reward_1 = 0


                next_s_1 = S_1[(agent_1_belief, curr_event)]
                agent_1_communicate = q_1[next_s_1]
                    
                if agent_1_communicate ==1:
                    communicate_count[0] += 1
                    (a1_communication_protocol[agent_1_belief, curr_event])[0] += 1
                else:
                    communicate_count[1] += 1
                    (a1_communication_protocol[agent_1_belief, curr_event])[1] += 1


                v_state, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                system_state, agent_1_belief, agent_2_belief = v_state
                                
                comm_cost, penalty = reward
                
                reward_1 += comm_cost
                
                curr_event=info['curr_event']
                
                s_1 = next_s_1
                
                
                       
            if curr_event=='b':

                agent_id=2
                
                if s_2 != -1 :
                    # cumulative_reward[1] += (GAMMA**t_2)*reward_2
                    cumulative_reward[1] += reward_2
                    # return_value[1] += reward_2
                    t_2+=1
                    reward_2 = 0
                

                next_s_2 = S_2[(agent_2_belief, curr_event)]
                agent_2_communicate = q_2[next_s_2]
                
                
                if agent_2_communicate ==1:                    
                    communicate_count[2] += 1
                    (a2_communication_protocol[agent_2_belief, curr_event])[0] += 1  
                else:
                    communicate_count[3] += 1
                    (a2_communication_protocol[agent_2_belief, curr_event])[1] += 1  
                
                
                v_state, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                
                system_state, agent_1_belief, agent_2_belief = v_state
                
                comm_cost, penalty = reward
                
                reward_2 += comm_cost
                                
                curr_event=info['curr_event']
                
                s_2 = next_s_2

        
        reward_2 += penalty
        reward_1 += penalty
    
        # cumulative_reward[0] += (GAMMA**t_1)*reward_1
        # cumulative_reward[1] += (GAMMA**t_2)*reward_2
        
        cumulative_reward[0] += reward_1
        cumulative_reward[1] += reward_2
        
        final_state = str(tuple((system_state, agent_1_belief, agent_2_belief)))

        T_state_dict[final_state] += 1     
    
    T_state_dict_list.append(T_state_dict)
    
    # print(dead_state_enter_count)
    communicate_counts.append(np.round(1/100*np.array(communicate_count),2))
    
    a1_protocol_list.append(a1_communication_protocol)
    a2_protocol_list.append(a2_communication_protocol)

    
    cumulative_reward[0] = np.round(cumulative_reward[0]/100, 2)
    cumulative_reward[1] = np.round(cumulative_reward[1]/100, 2)
    return_values_x.append(cumulative_reward[0])
    return_values_y.append(cumulative_reward[1])
    joint_return_values.append((cumulative_reward[0], cumulative_reward[1]))
    
# print(communicate_counts)

protocols_df["Agent 1 Average Cumulative Reward"] = return_values_x
protocols_df["Agent 2 Average Cumulative Reward"] = return_values_y

print(pd.DataFrame(joint_return_values, columns=['Agent 1 Return', 'Agent 2 Return']).value_counts())
 
plt.figure(figsize=(10,6))
plt.scatter(return_values_x[2:], return_values_y[2:], color='blue', label='Successful Protocols')
plt.scatter(return_values_x[0], return_values_y[0], color='red', label='Full Communication')
plt.scatter(return_values_x[1], return_values_y[1], color='green', label='No communication')

for i in range(len(return_values_x)):
    plt.text(return_values_x[i], return_values_y[i], f"{(return_values_x[i], return_values_y[i])}", fontsize=6, ha='left',va='bottom', color='purple', rotation=45)

plt.xlabel('Agent 1 Average Cumulative Reward per word')
plt.ylabel('Agent 2 Average Cumulative Reward per word')
# plt.title(' of Communication Protocols')
plt.legend()
plt.grid(True)
# plt.savefig(f'{FOLDER_NAME}/cumulative_reward_for_protocols.png')
plt.show()

communicate_counts = pd.DataFrame(communicate_counts, columns=['Agent 1 Communicate Count', 'Agent 1 Not Communicate Count', 'Agent 2 Communicate Count', 'Agent 2 Not Communicate Count'])

print(communicate_counts)

# print(T_state_dict_list)

T_state_df = pd.DataFrame(T_state_dict_list, columns=T_state_dict.keys())

print(T_state_df)

protocols_df = pd.concat([protocols_df, communicate_counts, T_state_df], axis=1)

protocols_df.to_csv(f'{FOLDER_NAME}/successful_protocols_with_stats.csv'   , index=False)

for i in range(len(a1_protocol_list)):
    a1_protocol = a1_protocol_list[i]
    a2_protocol = a2_protocol_list[i]
    
    
    print(f"\n================== {i}'th protocol ==================\nAgent 1 Communication Protocol:")
    for belief_state in a1_protocol:
        # print(belief_state)
        if (a1_protocol[belief_state] != [0,0]):
            print("In state ("+ str(m_bottom[belief_state[0]])+") Num Communicate: " + str(a1_protocol[belief_state][0]) + " Num Not Communicate: " + str(a1_protocol[belief_state][1]))

    print(f"\nAgent 2 Communication Protocol:")
    for belief_state in a2_protocol:
        if (a2_protocol[belief_state] != [0,0]):
            print("In state ("+ str(m_bottom[belief_state[0]])+") Num Communicate: " + str(a2_protocol[belief_state][0]) + " Num Not Communicate: " + str(a2_protocol[belief_state][1]))

