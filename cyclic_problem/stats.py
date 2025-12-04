import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from str_generator import RegexStringGenerator
import gymnasium as gym
import sys
sys.path.insert(0, './cyclic_problem')
import cyclic_problem_env

PHI_INV = {
    0:(False, 0 ),
    1:(False, 1 ),
    2:(False, 2 ),
    3:(False, 3 ),
    4:(False, 4 ),
    5:(False, 5 ),
    6:(False,-1 ),
    7:(True,  0 ),
    8:(True,  1 ),
    9:(True,  2 ),
    10:(True,  3 ),
    11:(True,  4 ),
    12:(True,  5 ),
    13:(True, -1 ),
}

PHI = {
    (False, 0 ):0,
    (False, 1 ):1,
    (False, 2 ):2,
    (False, 3 ):3,
    (False, 4 ):4,
    (False, 5 ):5,
    (False,-1 ):6,
    (True,  0 ):7,
    (True,  1 ):8,
    (True,  2 ):9,
    (True,  3 ):10,
    (True,  4 ):11,
    (True,  5 ):12,
    (True, -1 ):13,
}

# regexgen = RegexStringGenerator(max_star=5)

# string_list = []

# for i in range(1000):
#     string_list.append(regexgen.generate_training_str())
    
# df = pd.DataFrame(string_list, columns=["strings"])
# df.to_csv("cyclic_problem/strings.csv", index=False)


# df = pd.read_csv("cyclic_problem/strings.csv")
# string_list = df["strings"].to_list()


successful_protocols = pd.read_csv("cyclic_problem/simulation_2_successful_protocols.csv")
failed_protocols = pd.read_csv("cyclic_problem/simulation_2_failed_protocols.csv")

protocols_list = [successful_protocols, failed_protocols]
labels = ['Successful Protocols', 'Failed Protocols', ]
colors = ['blue', 'red']

plt.figure(figsize=(10,6))

for protocols, label, color in zip(protocols_list, labels, colors):
    return_values_x = []
    return_values_y = []
    for index, row in protocols.iterrows():
        protocol = row["Communication Protocols"].replace("(","").replace(")","").split(", ")
        protocol = [int(x) for x in protocol]
        q_1 = protocol[:7]
        q_2 = protocol[7:]
        
        return_value = [0,0]
        
        env = gym.make("CylicEnv-v0", render_mode = None, string_mode="stats")
        
        
        a_1_comm_count = 0
        a_2_comm_count = 0
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
            
            while not(terminated):
                if curr_symbol == "a":
                    
                    agent_id=1
                    agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
                    
                    if agent_1_prev_row_num != -1:
                        return_value[0] += (0.5**t_1)*reward_1
                        t_1+=1
                        reward_1 = 0
                    
                    if agent_2_in_dead_state:
                        agent_1_communicate = 0
                    else:
                        agent_1_communicate = q_1[agent_1_row_num]
                    
                    if agent_1_communicate ==1:
                        a_1_comm_count += 1
                    
                    config, reward, terminated, simulation_result, info = env.step((agent_id, agent_1_communicate))
                    
                    global_state, agent_1_belief, agent_2_belief = config
                    
                    reward_1 += reward
                    
                    curr_symbol=info['input_alphabet']
                    
                    agent_1_in_dead_state = agent_1_belief == -1
                    
                    agent_1_prev_row_num = agent_1_row_num
                    
                if curr_symbol == 'b': # curr_symbol == "c"
                    
                    agent_id=2
                    agent_2_row_num = PHI[(agent_1_in_dead_state, agent_2_belief)]
                    
                    if agent_2_prev_row_num != -1:
                        return_value[1] += (0.5**t_2)*reward_2
                        t_2+=1
                        reward_2 = 0
                    
                    if agent_1_in_dead_state:
                        agent_2_communicate = 0
                    else:
                        agent_2_communicate = q_2[agent_2_row_num]
                    
                    if agent_2_communicate ==1:
                        a_2_comm_count += 1
                    
                    config, reward, terminated, simulation_result, info = env.step((agent_id, agent_2_communicate))
                    global_state, agent_1_belief, agent_2_belief = config   
                    reward_2 += reward
                    curr_symbol=info['input_alphabet']
                    agent_2_in_dead_state = agent_2_belief == -1
                    agent_2_prev_row_num = agent_2_row_num
            
            reward_1 += reward
            reward_2 += reward
            
            return_value[0] += (0.5**t_1)*reward_1
            return_value[1] += (0.5**t_2)*reward_2
            # if i==250 or i==500 or i==750:
                # print(f"{a_1_comm_count, a_2_comm_count}, Intermediate Return Values: Agent 1: {np.round(return_value[0]/(i+1),2)}, Agent 2: {np.round(return_value[1]/(i+1),2)}")
        
        
        return_value[0] = np.round(return_value[0]/1000, 2)
        return_value[1] = np.round(return_value[1]/1000, 2)
        return_values_x.append(return_value[0])
        return_values_y.append(return_value[1])
    print(f"Average Communication Counts for {label}: Agent 1: {np.round(a_1_comm_count/1000, 2)}, Agent 2: {np.round(a_2_comm_count/1000, 2)}")
    
    plt.scatter(return_values_x, return_values_y, color=color, label=label)
    
    print(f"average return values for {label}: Agent 1: {np.round(np.mean(return_values_x),2)}, Agent 2: {np.round(np.mean(return_values_y),2)}")




plt.xlabel('Agent 1 Average Cumulative Rewards')
plt.ylabel('Agent 2 Average Cumulative Rewards')
plt.legend()
plt.grid(True)
plt.savefig("cyclic_problem/protocol_cumreward_values_scatter_plot.png")
plt.show()




