import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import uo_problem_env
from word_generator import WordGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from uo_problem_q import FOLDER_NAME

file_name = "exp3_a_0.01_g_0.1_e_0.1_with_stats"

protocols_df = pd.read_csv(f'{FOLDER_NAME}/{file_name}.csv')

# print(protocols_df.columns[14:])

protocol = protocols_df[protocols_df.columns[14:]]

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
    
    mask = protocols_df[unique_protocols.columns].eq(row[1:]).all(axis=1)
    
    mask_df = protocols_df[mask]
    
    count.append(mask_df['Success Count'].sum())
    
    print(f"Protocol: {list(zip(unique_protocols.columns, row[1:]))}, count: {mask_df['Success Count'].sum()}")
    
    
print(np.sum(count))