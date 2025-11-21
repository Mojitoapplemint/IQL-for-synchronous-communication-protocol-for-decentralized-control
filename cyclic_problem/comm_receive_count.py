import numpy as np
import gymnasium as gym
import pandas as pd
import random
import cyclic_problem_env

# q_1 = pd.read_csv("./cyclic_problem/demo_q1_table.csv")
# q_2 = pd.read_csv("./cyclic_problem/demo_q2_table.csv")
# q_1 = q_1.drop(q_1.columns[[0]], axis=1).to_numpy()
# q_2 = q_2.drop(q_2.columns[[0]], axis=1).to_numpy()

PHI = {
    (False, 0 ):0,
    (False, 1 ):1,
    (False, 2 ):2,
    (False, 3 ):3,
    (False, 4 ):4,
    (False, 5 ):5,
    (False,-1 ):6,
    (True,  0):7,
    (True,  1):8,
    (True,  2):9,
    (True,  3):10,
    (True,  4):11,
    (True,  5):12,
    (True, -1):13,
}

string_mode = "simulation" # options: "simulation", "training

env = gym.make("CylicEnv-v0", render_mode = None, string_mode=string_mode)

test_count = 1000


a_1_receive = [[0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0]]

a_2_receive = [[0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0]]

successful_protocols = pd.read_csv("cyclic_problem/simulation_2_successful_protocols.csv")

a_1_receive_percentage = [[],[],[],[],[],[],[]]
a_2_receive_percentage = [[],[],[],[],[],[],[]]

for index, row in successful_protocols.iterrows():
    print(f"Simulation run {index+1}/{len(successful_protocols)}", end="\r")
    protocol = row["Communication Protocols"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    q_1 = protocol[:7]
    q_2 = protocol[7:]

    for i in range(test_count):
        terminated = False
        simulation_result = False

        config, info = env.reset()

        global_state, agent_1_belief, agent_2_belief = config

        curr_symbol=info['input_alphabet']
        string = info["string"]

        agent_1_in_dead_state = False
        agent_2_in_dead_state = False

        while not(terminated):
            if curr_symbol == "a":
                
                agent_id=1
                agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
                
                if agent_2_in_dead_state:
                    agent_1_communicate = 0
                else:
                    agent_1_communicate = q_1[agent_1_row_num]
                    
                if agent_1_communicate == 1:
                    a_2_receive[agent_2_belief][1] += 1
                else:
                    a_2_receive[agent_2_belief][0] += 1
                
                config, reward, terminated, simulation_result, info = env.step((agent_id, agent_1_communicate))
                
                global_state, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                
                curr_symbol=info['input_alphabet']
                            
            if curr_symbol == "b":
                agent_id=2
                agent_2_row_num = PHI[(agent_1_in_dead_state, agent_2_belief)]
                
                if agent_1_in_dead_state:
                    agent_2_communicate = 0
                else:        
                    agent_2_communicate = q_2[agent_2_row_num]
                
                if agent_2_communicate == 1:
                    a_1_receive[agent_1_belief][1] += 1
                else:
                    a_1_receive[agent_1_belief][0] += 1    
                
                config, reward, terminated, simulation_result, info = env.step((agent_id, agent_2_communicate))
                
                global_state, agent_1_belief, agent_2_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                curr_symbol=info['input_alphabet']
    for belief_state in range(7):
        if (a_1_receive[belief_state][0]+a_1_receive[belief_state][1])==0:
            a_1_receive_percentage[belief_state].append(0)
        else:
            a_1_receive_percentage[belief_state].append(a_1_receive[belief_state][1]/(a_1_receive[belief_state][0]+a_1_receive[belief_state][1])*100)
        
        if (a_2_receive[belief_state][0]+a_2_receive[belief_state][1])==0:
            a_2_receive_percentage[belief_state].append(0)
        else:
            a_2_receive_percentage[belief_state].append(a_2_receive[belief_state][1]/(a_2_receive[belief_state][0]+a_2_receive[belief_state][1])*100)

print("Agent 1 received communication 'b' counts per belief state:")
for belief_state in range(7):
    # if (a_1_receive[belief_state][0]+a_1_receive[belief_state][1])==0:
    #     print("   In belief state", belief_state, " agent 1 received communication in percentage of 0 %")
    # else:
        # print("   In belief state", belief_state, " agent 1 received communication in percentage of ", np.round(a_1_receive[belief_state][1]/(a_1_receive[belief_state][0]+a_1_receive[belief_state][1])*100,2) , "%")
    # print("   Belief state", belief_state, ": No comm =", when_agent_1_receive_comm[belief_state][0]/len(successful_protocols), ", Comm =", when_agent_1_receive_comm[belief_state][1]/len(successful_protocols))

    print("   Belief state", belief_state, ": Average percentage of receiving 'b' communication =", np.round(np.mean(a_1_receive_percentage[belief_state]),2) , "%")

print(f"\nAgent 2 received communication 'a' counts per belief state:")
for belief_state in range(7):
    # if (a_2_receive[belief_state][0]+a_2_receive[belief_state][1])==0:
    #     print("   In belief state", belief_state, " agent 2 received communication in percentage of 0 %")
    # else:
        # print("   In belief state", belief_state, " agent 2 received communication in percentage of ", np.round(a_2_receive[belief_state][1]/(a_2_receive[belief_state][0]+a_2_receive[belief_state][1])*100, 2) , "%")
    # print("   Belief state", belief_state, ": No comm =", when_agent_2_receive_comm[belief_state][0]/len(successful_protocols), ", Comm =", when_agent_2_receive_comm[belief_state][1]/len(successful_protocols))
    print("   Belief state", belief_state, ": Average percentage of receiving 'a' communication =", np.round(np.mean(a_2_receive_percentage[belief_state]),2) , "%")