import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import uo_problem_env
from word_generator import WordGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import seaborn as sns
from uo_problem_q import S_1, S_2, A1_OBS, A2_OBS, FOLDER_NAME

m_L_bot={
    1:  "{1,3}",
    2:  "{2,4,5,12,20,17}",
    3:  "{6,21}",
    4:  "{6,10}",
    5:  "{2,4,8,13,18}",
    6:  "{6}",
    7:  "{7}",
    8:  "{2,4,14}",
    9:  "{9}",
    10: "{10}",
    11: "{11}",
    12: "{2,4}",
    13: "{5,12,17,20}",
    14: "{14}",
    15: "{15}",
    16: "{16}",
    17: "{8,13,18}",
    19: "{19}",
    21: "{21}",
    -1: "{⊥}",
}

exp_number = 3

# file_name = f"successful_protocols_exp{exp_number}"
file_name = "baselines"

successful_protocols = pd.read_csv(f'{FOLDER_NAME}/{file_name}.csv')

GAMMA = 0.1

success_return_values_x = []
success_return_values_y = []
joint_return_values = []

communication_counts = []

a1_protocol_list= []
a2_protocol_list= []

T_state_dict_list = []

test_count = 300

for index, row in successful_protocols.iterrows():
    print(f"{index} / {len(successful_protocols)}", end="\r")
    
    protocol = row["Protocol"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    q_1 = protocol[:40]
    q_2 = protocol[40:]
        
    return_value = [0,0]
    
    env = gym.make("UOEnv-v1", render_mode = None, string_mode="simulation")
    
    communication_count = [0,0,0,0]
    
    comm_dict_in_dead_state={}
    comm_dict_not_in_dead_state={}
    
    T_state_dict={
        (9,-1,-1):0,
        (7,7,-1):0,
        (11,11,-1):0,
        (11,11,3):0,
        (11,-1,-1):0,
        (11,11,11):0,
        (7,7,7):0,
        (7,12,7):0,
        (7,7,3):0,
        (7,7,6):0,
    }
    
    a1_communication_protocol={
        i: [0,0] for i in list(S_1.keys())[:40]
    }
    
    a2_communication_protocol={
        i: [0,0] for i in list(S_2.keys())[:120]
    }
    
    for i in range (test_count):
        terminated = False
        simulation_result = False


        v_state, info = env.reset()

        system_state, agent_1_belief, agent_2_belief = v_state

        curr_event=info['curr_event']
        
        word = info['word']

        s_1 = -1
        s_2 = -1

        reward_1=0
        reward_2=0
        
        t_1=0
        t_2=0        
        while not (terminated):

            if curr_event in A1_OBS:
                
                agent_id=1
                
                if s_1 != -1 :
                    return_value[0] += (GAMMA**t_1)*reward_1
                    # return_value[0] += reward_1
                    t_1+=1
                    reward_1 = 0


                s_1 = S_1[(agent_1_belief, curr_event)]
                agent_1_communicate = q_1[s_1]
                
                if agent_1_communicate ==1:
                    communication_count[0] += 1
                    (a1_communication_protocol[agent_1_belief, curr_event])[0] += 1
                else:
                    communication_count[1] += 1
                    (a1_communication_protocol[agent_1_belief, curr_event])[1] += 1

                v_state, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                system_state, agent_1_belief, agent_2_belief = v_state
                                
                comm_cost, penalty = reward
                
                reward_1 += comm_cost

                            
            if curr_event in A2_OBS:

                agent_id=2
                
                if s_2 != -1 :
                    return_value[1] += (GAMMA**t_2)*reward_2
                    # return_value[1] += reward_2
                    t_2+=1
                    reward_2 = 0
                

                s_2 = S_2[(agent_2_belief, curr_event)]
                agent_2_communicate = q_2[s_2]
            
                if agent_2_communicate ==1:    
                    communication_count[2] += 1
                    (a2_communication_protocol[agent_2_belief, curr_event])[0] += 1  
                else:
                    communication_count[3] += 1
                    (a2_communication_protocol[agent_2_belief, curr_event])[1] += 1  
            
                v_state, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                
                system_state, agent_1_belief, agent_2_belief = v_state
                
                comm_cost, penalty = reward
                
                reward_2 += comm_cost            
            
            agent_2_in_dead_state = agent_2_belief == -1
            
            agent_1_in_dead_state = agent_1_belief == -1
            
            curr_event=info['curr_event']

        T_state_dict[tuple(v_state)]+=1
        
        reward_2 += penalty
        reward_1 += penalty
        
        # return_value[0] += reward_1
        # return_value[1] += reward_2
        return_value[0] += (GAMMA**t_1)*reward_1
        return_value[1] += (GAMMA**t_2)*reward_2
    
        
    T_state_dict_list.append(T_state_dict)

    
    a1_protocol_list.append(a1_communication_protocol)
    a2_protocol_list.append(a2_communication_protocol)
    
    communication_counts.append(1/test_count*np.array(communication_count))
    
    return_value[0] = return_value[0]/test_count
    return_value[1] = return_value[1]/test_count
    success_return_values_x.append(return_value[0])
    success_return_values_y.append(return_value[1])
    joint_return_values.append((return_value[0], return_value[1]))

# successful_protocols["Agent 1 Average Cumulative Reward"] = success_return_values_x
# successful_protocols["Agent 2 Average Cumulative Reward"] = success_return_values_y

# successful_protocols.to_csv("problem_w_unobservable_events/larger_11_penalty_successful_protocols_with_returns.csv", index=False)

joint_return_values_df = pd.DataFrame(joint_return_values, columns=['A1 Return', 'A2 Return'])

# print(joint_return_values_df.value_counts())

communication_counts = pd.DataFrame(communication_counts, columns=['A1 Com', 'A1 Not Com', 'A2 Com', 'A2 Not Com'])

print(communication_counts)

# print(T_state_dict)

T_state_df = pd.DataFrame(T_state_dict_list, columns=T_state_dict.keys())


a1_protocol_df = pd.DataFrame(a1_protocol_list)
a2_protocol_df = pd.DataFrame(a2_protocol_list)

for column in a1_protocol_df.columns:    
    if np.sum(np.sum(a1_protocol_df[column])) == 0:
        a1_protocol_df=a1_protocol_df.drop(columns=[column])

for column in a2_protocol_df.columns:    
    if np.sum(np.sum(a2_protocol_df[column])) == 0:
        a2_protocol_df=a2_protocol_df.drop(columns=[column])


protocols_df = pd.concat([successful_protocols, joint_return_values_df, communication_counts, T_state_df, a1_protocol_df, a2_protocol_df], axis=1)

return_data = []


for return_value in joint_return_values_df["A1 Return"].unique():
    
    mask_df = protocols_df[protocols_df['A1 Return'] == return_value]
    
    count = mask_df['Success Count'].sum()
    
    print(f"Agent 1 Return : {return_value}, count: {count}")
    
    return_data.append((return_value, count))

return_data_df = pd.DataFrame(return_data, columns=['A1 Return', 'Count'])

return_weighted_mean = np.sum(return_data_df['A1 Return'] * return_data_df['Count']) / np.sum(return_data_df['Count'])

print("Weighted Mean of Agent 1 Return:", f"{return_weighted_mean:.3f}", "\n")

return_data = []

for return_value in joint_return_values_df["A2 Return"].unique():
    
    mask_df = protocols_df[protocols_df['A2 Return'] == return_value]
    
    count = mask_df['Success Count'].sum()
    
    print(f"Agent 2 Return : {return_value}, count: {count}")
    
    return_data.append((return_value, count))


return_data_df = pd.DataFrame(return_data, columns=['A2 Return', 'Count'])

return_weighted_mean = np.sum(return_data_df['A2 Return'] * return_data_df['Count']) / np.sum(return_data_df['Count'])
print("Weighted Mean of Agent 2 Return:", f"{return_weighted_mean:.3f}", "\n")

com_data = []

for avg_com in communication_counts["A1 Com"].unique():
    
    mask_df = protocols_df[protocols_df['A1 Com'] == avg_com]
    
    count = mask_df['Success Count'].sum()
    
    com_data.append((avg_com, count))

com_data_df = pd.DataFrame(com_data, columns=['A1 Com', 'Count'])

communication_weighted_mean = np.sum(com_data_df['A1 Com'] * com_data_df['Count']) / np.sum(com_data_df['Count'])
    
print("Weighted Mean of Agent 1 Communication:", f"{communication_weighted_mean:.3f}", "\n")

com_data = []

for avg_com in communication_counts["A2 Com"].unique():
    
    mask_df = protocols_df[protocols_df['A2 Com'] == avg_com]
    
    count = mask_df['Success Count'].sum()
    
    com_data.append((avg_com, count))

com_data_df = pd.DataFrame(com_data, columns=['A2 Com', 'Count'])

communication_weighted_mean = np.sum(com_data_df['A2 Com'] * com_data_df['Count']) / np.sum(com_data_df['Count'])
    
print("Weighted Mean of Agent 2 Communication:", f"{communication_weighted_mean:.3f}", "\n")

for T_state in T_state_dict.keys():
    visit_data = []
    for num_visit in T_state_df[T_state].unique():
        mask_df = protocols_df[protocols_df[T_state] == num_visit]
        count = mask_df['Success Count'].sum()
        visit_data.append((num_visit, count))
    visit_data_df = pd.DataFrame(visit_data, columns=[f'Visit Count of {T_state}', 'Count'])
    weighted_mean_visit = np.sum(visit_data_df[f'Visit Count of {T_state}'] * visit_data_df['Count']) / np.sum(visit_data_df['Count'])
    print(f"Weighted Mean of Visit Count for {T_state}: {weighted_mean_visit:.3f}")

# print(return_data_df)


protocols_df.to_csv(f"{FOLDER_NAME}/{file_name}_with_stats.csv", index=False)

# plt.figure(figsize=(10,6))
# ax = sns.barplot(x=return_data_df['A2 Return'], y=return_data_df['Count'], alpha=0.7, color='blue', edgecolor='black')
# plt.xlabel('Agent 2 Average Return')
# # plt.title(f'Exp {exp_number} Return Values for Agent 2 / Weighted Mean {return_weighted_mean:.3f}')
# plt.legend()
# for c in ax.containers:
#     ax.bar_label(c) # fmt='%d' ensures the labels are displayed as integers

# plt.savefig(f'{FOLDER_NAME}/Exp {exp_number} Return.png')
# plt.show()
