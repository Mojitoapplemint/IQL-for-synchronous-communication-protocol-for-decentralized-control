import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import uo_problem_env
from word_generator import WordGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from uo_problem_q import FOLDER_NAME

exp_number = 3

file_name = f"successful_protocols_exp{exp_number}"
# file_name = "baselines"

protocol_df = pd.read_csv(f'{FOLDER_NAME}/{file_name}_with_stats.csv')

# print(protocols_df.columns[14:])

protocol = protocol_df[protocol_df.columns[20:]]

unique_protocols = protocol.drop_duplicates()

pd.set_option('display.max_columns', None)

# print(unique_protocols)

for column in unique_protocols.columns:
    count = 0
    for row in unique_protocols[column]:
        if row[1] != '0':
            count += 1
    
    if count == 0:
        unique_protocols = unique_protocols.drop(columns=[column])


print(unique_protocols)

count = []
for row in unique_protocols.itertuples():
    
    mask = protocol_df[unique_protocols.columns].eq(row[1:]).all(axis=1)
    
    mask_df = protocol_df[mask]
    
    count.append(mask_df['Success Count'].sum())
    
    print(f"Protocol: {list(zip(unique_protocols.columns, row[1:]))}, count: {mask_df['Success Count'].sum()}")
    
    
print(np.sum(count))