import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import sys
sys.path.insert(0, './problem_w_unobservable_events')
import uo_problem_env
from str_generator import StringGenerator

A1_OBSERVABLE_EVENTS = ['a', 'c']
A2_OBSERVABLE_EVENTS = ['x', 'y', 'z', 's', 't', 'r']

PHI_1={
    (1, 'a'):0,
    (2, 'a'):1,
    (3, 'a'):2,
    (4, 'a'):3,
    (5, 'a'):4,
    (6, 'a'):5,
    (7, 'a'):6,
    (8, 'a'):7,
    (9, 'a'):8,
    (10,'a'):9,
    (11,'a'):10,
    (12,'a'):11,
    (13,'a'):12,
    (14,'a'):13,
    (15,'a'):14,
    (16,'a'):15,
    (17,'a'):16,
    (19,'a'):17,
    (21,'a'):18,
    (-1,'a'):19,
    (1, 'c'):20,
    (2, 'c'):21,
    (3, 'c'):22,
    (4, 'c'):23,
    (5, 'c'):24,
    (6, 'c'):25,
    (7, 'c'):26,
    (8, 'c'):27,
    (9, 'c'):28,
    (10,'c'):29,
    (11,'c'):30,
    (12,'c'):31,
    (13,'c'):32,
    (14,'c'):33,
    (15,'c'):34,
    (16,'c'):35,
    (17,'c'):36,
    (19,'c'):37,
    (21,'c'):38,
    (-1,'c'):39,
}

PHI_2={
    (1, 'x'):0,
    (2, 'x'):1,
    (3, 'x'):2,
    (4, 'x'):3,
    (5, 'x'):4,
    (6, 'x'):5,
    (7, 'x'):6,
    (8, 'x'):7,
    (9, 'x'):8,
    (10,'x'):9,
    (11,'x'):10,
    (12,'x'):11,
    (13,'x'):12,
    (14,'x'):13,
    (15,'x'):14,
    (16,'x'):15,
    (17,'x'):16,
    (19,'x'):17,
    (21,'x'):18,
    (-1,'x'):19,
    (1, 'y'):20,
    (2, 'y'):21,
    (3, 'y'):22,
    (4, 'y'):23,
    (5, 'y'):24,
    (6, 'y'):25,
    (7, 'y'):26,
    (8, 'y'):27,
    (9, 'y'):28,
    (10,'y'):29,
    (11,'y'):30,
    (12,'y'):31,
    (13,'y'):32,
    (14,'y'):33,
    (15,'y'):34,
    (16,'y'):35,
    (17,'y'):36,
    (19,'y'):37,
    (21,'y'):38,
    (-1,'y'):39,
    (1, 'z'):40,
    (2, 'z'):41,
    (3, 'z'):42,
    (4, 'z'):43,
    (5, 'z'):44,
    (6, 'z'):45,
    (7, 'z'):46,
    (8, 'z'):47,
    (9, 'z'):48,
    (10,'z'):49,
    (11,'z'):50,
    (12,'z'):51,
    (13,'z'):52,
    (14,'z'):53,
    (15,'z'):54,
    (16,'z'):55,
    (17,'z'):56,
    (19,'z'):57,
    (21,'z'):58,
    (-1,'z'):59,
    (1, 's'):60,
    (2, 's'):61,
    (3, 's'):62,
    (4, 's'):63,
    (5, 's'):64,
    (6, 's'):65,
    (7, 's'):66,
    (8, 's'):67,
    (9, 's'):68,
    (10,'s'):69,
    (11,'s'):70,
    (12,'s'):71,
    (13,'s'):72,
    (14,'s'):73,
    (15,'s'):74,
    (16,'s'):75,
    (17,'s'):76,
    (19,'s'):77,
    (21,'s'):78,
    (-1,'s'):79,
    (1, 't'):80,
    (2, 't'):81,
    (3, 't'):82,
    (4, 't'):83,
    (5, 't'):84,
    (6, 't'):85,
    (7, 't'):86,
    (8, 't'):87,
    (9, 't'):88,
    (10,'t'):89,
    (11,'t'):90,
    (12,'t'):91,
    (13,'t'):92,
    (14,'t'):93,
    (15,'t'):94,
    (16,'t'):95,
    (17,'t'):96,
    (19,'t'):97,
    (21,'t'):98,
    (-1,'t'):99,
    (1, 'r'):100,
    (2, 'r'):101,
    (3, 'r'):102,
    (4, 'r'):103,
    (5, 'r'):104,
    (6, 'r'):105,
    (7, 'r'):106,
    (8, 'r'):107,
    (9, 'r'):108,
    (10,'r'):109,
    (11,'r'):110,
    (12,'r'):111,
    (13,'r'):112,
    (14,'r'):113,
    (15,'r'):114,
    (16,'r'):115,
    (17,'r'):116,
    (19,'r'):117,
    (21,'r'):118,
    (-1,'r'):119,
}

m_bottom={
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
    -1: "{-1}",
}

# regexgen = StringGenerator(max_star=5)

# string_list = []

# string_list = regexgen.generate_stats_str()

# df = pd.DataFrame(string_list, columns=["strings"])
# df.to_csv("problem_w_unobservable_events/strings.csv", index=False)
    

# df = pd.read_csv("problem_w_unobservable_events/strings.csv")
# string_list = df["strings"].to_list()


successful_protocols = pd.read_csv("problem_w_unobservable_events/exp_successful_protocols.csv")

success_return_values_x = []
success_return_values_y = []
joint_return_values = []

return_values = [0,0]
communication_counts = []

a1_protocol_list= []
a2_protocol_list= []

communication_dict={}

for index, row in successful_protocols.iterrows():
    print(f"{index} / {len(successful_protocols)}", end="\r")
    
    protocol = row["Communication Protocols"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    q_1 = protocol[:40] + [0 for _ in range(40)]
    q_2 = protocol[40:] + [0 for _ in range(120)]
    
    return_value = [0,0]
    
    env = gym.make("UOEnv-v0", render_mode = None, string_mode="stats")
    
    communication_count = [0,0,0,0]
    
    a1_communication_protocol={
            1 :[0,0,0,0],
            2 :[0,0,0,0],
            3 :[0,0,0,0],
            4 :[0,0,0,0],
            5 :[0,0,0,0],
            6 :[0,0,0,0],
            7 :[0,0,0,0],
            8 :[0,0,0,0],
            9 :[0,0,0,0],
            10:[0,0,0,0],
            11:[0,0,0,0],
            12:[0,0,0,0],
            13:[0,0,0,0],
            14:[0,0,0,0],
            15:[0,0,0,0],
            16:[0,0,0,0],
            17:[0,0,0,0],
            19:[0,0,0,0],
            21:[0,0,0,0],
            -1:[0,0,0,0],
    }
    
    a2_communication_protocol={
            1 :[0,0,0,0,0,0,0,0,0,0,0,0],
            2 :[0,0,0,0,0,0,0,0,0,0,0,0],
            3 :[0,0,0,0,0,0,0,0,0,0,0,0],
            4 :[0,0,0,0,0,0,0,0,0,0,0,0],
            5 :[0,0,0,0,0,0,0,0,0,0,0,0],
            6 :[0,0,0,0,0,0,0,0,0,0,0,0],
            7 :[0,0,0,0,0,0,0,0,0,0,0,0],
            8 :[0,0,0,0,0,0,0,0,0,0,0,0],
            9 :[0,0,0,0,0,0,0,0,0,0,0,0],
            10:[0,0,0,0,0,0,0,0,0,0,0,0],
            11:[0,0,0,0,0,0,0,0,0,0,0,0],
            12:[0,0,0,0,0,0,0,0,0,0,0,0],
            13:[0,0,0,0,0,0,0,0,0,0,0,0],
            14:[0,0,0,0,0,0,0,0,0,0,0,0],
            15:[0,0,0,0,0,0,0,0,0,0,0,0],
            16:[0,0,0,0,0,0,0,0,0,0,0,0],
            17:[0,0,0,0,0,0,0,0,0,0,0,0],
            19:[0,0,0,0,0,0,0,0,0,0,0,0],
            21:[0,0,0,0,0,0,0,0,0,0,0,0],
            -1:[0,0,0,0,0,0,0,0,0,0,0,0],
    }
    
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
                    return_value[0] += (0.1**t_1)*reward_1
                    # return_value[0] += reward_1
                    t_1+=1
                    reward_1 = 0

                if agent_2_in_dead_state:
                    agent_1_communicate = 0
                else:
                    agent_1_row_num = len(PHI_1)+PHI_1[(agent_1_belief, curr_symbol)] if agent_2_in_dead_state else PHI_1[(agent_1_belief, curr_symbol)]
                    agent_1_communicate = q_1[agent_1_row_num]
                    
                if agent_1_communicate ==1:
                    communication_count[0] += 1 
                    a1_communication_protocol[agent_1_belief][A1_OBSERVABLE_EVENTS.index(curr_symbol)*2] += 1
                else:
                    communication_count[1] += 1
                    a1_communication_protocol[agent_1_belief][A1_OBSERVABLE_EVENTS.index(curr_symbol)*2+1] += 1

                
                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                    
                reward_1 += reward
                
                curr_symbol=info['input_alphabet']
                
                agent_1_prev_row_num = agent_1_row_num
                            
            if curr_symbol in ['x', 'y', 'z', 's', 't', 'r']:

                agent_id=2
                
                if agent_2_prev_row_num != -1 :
                    return_value[1] += (0.1**t_2)*reward_2
                    # return_value[1] += reward_2
                    t_2+=1
                    reward_2 = 0
                
                if agent_1_in_dead_state:
                    agent_2_communicate = 0
                else:
                    agent_2_row_num = len(PHI_2)+PHI_2[(agent_2_belief, curr_symbol)] if agent_1_in_dead_state else PHI_2[(agent_2_belief, curr_symbol)]
                    agent_2_communicate = q_2[agent_2_row_num]
                
                if curr_symbol not in communication_dict:
                    communication_dict[curr_symbol] = [0,0]
                
                
                if agent_2_communicate ==1:
                    communication_count[2] += 1
                    communication_dict[curr_symbol][1] += 1
                    a2_communication_protocol[agent_2_belief][A2_OBSERVABLE_EVENTS.index(curr_symbol)*2] += 1
                else:
                    communication_count[3] += 1
                    communication_dict[curr_symbol][0] += 1
                    a2_communication_protocol[agent_2_belief][A2_OBSERVABLE_EVENTS.index(curr_symbol)*2+1] += 1
                
                
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
        return_value[0] += (0.1**t_1)*reward_1
        return_value[1] += (0.1**t_2)*reward_2
        
        if communication_count[0]==0 and communication_count[1]==0:
            print(protocol)
    
    a1_protocol_list.append(a1_communication_protocol)
    a2_protocol_list.append(a2_communication_protocol)
    
    communication_counts.append(communication_count)
    
    return_value[0] = np.round(return_value[0]/1000, 2)
    return_value[1] = np.round(return_value[1]/1000, 2)
    success_return_values_x.append(return_value[0])
    success_return_values_y.append(return_value[1])
    joint_return_values.append((return_value[0], return_value[1]))

print(pd.DataFrame(joint_return_values, columns=['Agent 1 Return', 'Agent 2 Return']).value_counts())
 
plt.figure(figsize=(10,6))
plt.scatter(success_return_values_x, success_return_values_y, color='blue', label='Successful Protocols')
plt.xlabel('Agent 1 Average Return')
plt.ylabel('Agent 2 Average Return')
plt.title('Return Values of Communication Protocols')
plt.legend()
plt.grid(True)
plt.savefig('problem_w_unobservable_events/exp_stats_successful_protocols_returns.png')
plt.show()

communication_counts = pd.DataFrame(communication_counts, columns=['Agent 1 Communicate', 'Agent 1 Not Communicate', 'Agent 2 Communicate', 'Agent 2 Not Communicate'])

print(communication_counts)

print(communication_dict)

successful_protocols = pd.read_csv("problem_w_unobservable_events/exp_2_successful_protocols.csv")

