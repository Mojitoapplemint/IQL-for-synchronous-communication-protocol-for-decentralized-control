import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from word_generator import RegexWordGenerator
import gymnasium as gym
import cyclic_problem_env

GAMMA = 0.1

PHI = {
    (False, 0 ):0,
    (False, 1 ):1,
    (False, 2 ):2,
    (False, 3 ):3,
    (False, 4 ):4,
    (False, 5 ):5,
    (False, 6 ):6,
    (False, 7 ):7,
    (False,-1 ):8,
    (True, 0 ):9,
    (True, 1 ):10,
    (True, 2 ):11,
    (True, 3 ):12,
    (True, 4 ):13,
    (True, 5 ):14,
    (True, 6 ):15,
    (True, 7 ):16,
    (True,-1 ):17,
}


# regexgen = RegexWordGenerator(max_star=5)

# string_list = []

# for i in range(1000):
#     string_list.append(regexgen.generate_training_word())
    
# df = pd.DataFrame(string_list, columns=["strings"])
# df.to_csv("second_cyclic_problem/strings.csv", index=False)


successful_protocols = pd.read_csv("second_cyclic_problem/simulation_2_successful_protocols.csv")


success_return_values_x = []
success_return_values_y = []
joint_return_values = []

communicate_counts = []

communication_dict={}

for index, row in successful_protocols.iterrows():
    print(f"{index} / {len(successful_protocols)}", end="\r")
    protocol = row["Communication Protocols"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    q_1 = protocol[:40] + [0 for _ in range(40)]
    q_2 = protocol[40:] + [0 for _ in range(120)]
    
    return_value = [0,0]
    
    env = gym.make("CyclicEnv2-v0", render_mode = None, string_mode="stats")
    
    communicate_count = [0,0]
    
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
            if curr_symbol in ['a', 'c']:
                
                agent_id=1
                
                if agent_1_prev_row_num != -1 :
                    return_value[0] += (GAMMA**t_1)*reward_1
                    # return_value[0] += reward_1
                    t_1+=1
                    reward_1 = 0

                if agent_2_in_dead_state:
                    agent_1_communicate = 0
                else:
                    agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
                    agent_1_communicate = q_1[agent_1_row_num]
                    
                if agent_1_communicate ==1:
                    communicate_count[0] += 1 

                
                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                    
                reward_1 += reward
                
                curr_symbol=info['input_alphabet']
                
                agent_1_prev_row_num = agent_1_row_num
                            
            if curr_symbol in ['x', 'y', 'z', 's', 't', 'r']:

                agent_id=2
                
                if agent_2_prev_row_num != -1 :
                    return_value[1] += (GAMMA**t_2)*reward_2
                    # return_value[1] += reward_2
                    t_2+=1
                    reward_2 = 0
                
                if agent_1_in_dead_state:
                    agent_2_communicate = 0
                else:
                    agent_2_row_num = PHI[(agent_1_in_dead_state, agent_1_belief)]
                    agent_2_communicate = q_2[agent_2_row_num]
                
                if curr_symbol not in communication_dict:
                    communication_dict[curr_symbol] = [0,0]
                
                
                if agent_2_communicate ==1:
                    communicate_count[1] += 1
                    communication_dict[curr_symbol][1] += 1
                else:
                    communication_dict[curr_symbol][0] += 1

                
                
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                reward_2 += reward
                                
                curr_symbol=info['input_alphabet']
                
                agent_2_prev_row_num = agent_2_row_num

        
        reward_1 += reward
        reward_2 += reward
        
        # return_value[0] += reward_1
        # return_value[1] += reward_2
        return_value[0] += (GAMMA**t_1)*reward_1
        return_value[1] += (GAMMA**t_2)*reward_2
        
        if communicate_count[0]==0 and communicate_count[1]==0:
            print(protocol)
        
    communicate_counts.append(communicate_count)
    
    return_value[0] = np.round(return_value[0]/1000, 2)
    return_value[1] = np.round(return_value[1]/1000, 2)
    success_return_values_x.append(return_value[0])
    success_return_values_y.append(return_value[1])
    joint_return_values.append((return_value[0], return_value[1]))
    
# print(communicate_counts)

print(pd.DataFrame(joint_return_values, columns=['Agent 1 Return', 'Agent 2 Return']).value_counts())
 
plt.figure(figsize=(10,6))
plt.scatter(success_return_values_x, success_return_values_y, color='blue', label='Successful Protocols')
plt.xlabel('Agent 1 Average Return')
plt.ylabel('Agent 2 Average Return')
plt.title('Return Values of Communication Protocols')
plt.legend()
plt.grid(True)
plt.savefig('second_cyclic_problem/stats_successful_protocols_returns.png')
plt.show()

communicate_counts = pd.DataFrame(communicate_counts, columns=['Agent 1 Communicate Count', 'Agent 2 Communicate Count'])

print(communicate_counts.value_counts())

print(communication_dict)