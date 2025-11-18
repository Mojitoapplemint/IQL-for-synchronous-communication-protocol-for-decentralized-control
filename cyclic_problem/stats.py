import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

q_1_protocols = {
    0:0,
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
    6:0,
    7:0,
    8:0,
    9:0,
    10:0,
    11:0,
    12:0,
    13:0
}

q_2_protocols = {
    0:0,
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
    6:0,
    7:0,
    8:0,
    9:0,
    10:0,
    11:0,
    12:0,
    13:0
}

# df = pd.read_csv("./cyclic_problem/simulation_2_results.csv")[1:]

# plt.bar(df['Fail Rate (%)'], df['Count'])
# plt.xlabel('Fail Rate (%)')
# plt.ylabel('Count')
# plt.title('Fail Rate Distribution over 10000 Simulations (excluding 0% fail rate)')
# plt.show()

# Analyze successful protocols
success_df = pd.read_csv("./cyclic_problem/simulation_2_successful_protocols.csv")

num_successful = success_df['Success Count'].sum()

for index, row in success_df.iterrows():
    protocol_key = eval(row['Communication Protocols'])
    q1_protocol = protocol_key[0]
    q2_protocol = protocol_key[1]
    
    for i in range(13):
        q_1_protocols[i] += q1_protocol[i] * row['Success Count']
        q_2_protocols[i] += q2_protocol[i] * row['Success Count']

print("Aggregated Communication Protocol for Agent 1:")
for key in q_1_protocols:
    if key == 0:
        print("   When the agent 2 is not in dead state")
    elif key ==7:
        print("\n   When the aegnt 2 is in dead state")
    
    agent_2_in_dead_state, agent_1_belief = PHI_INV[key]
    
    
    print(f"    - Communicated 'a' {100*np.round(q_1_protocols[key]/num_successful, 2)}% of the time,", end=" ")
    print(f"when b1 = {agent_1_belief}\n", end="")
    
    


print("\n\nAggregated Communication Protocol for Agent 2:")
for key in q_2_protocols:
    if key == 0:
        print("   When the agent 1 is not in dead state")
    elif key ==7:
        print("\n   When the aegnt 1 is in dead state")
    agent_1_in_dead_state, agent_2_belief = PHI_INV[key]
    

    print(f"    - Communicated 'b' {100*np.round(q_2_protocols[key]/num_successful, 2)}% of the time,", end=" ")
    print(f"when b2 = {agent_2_belief}\n", end="")