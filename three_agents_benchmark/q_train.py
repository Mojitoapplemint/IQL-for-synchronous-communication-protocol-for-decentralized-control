
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import three_agents_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agent_q import q_training, FOLDER_NAME

env = gym.make('ThreeAgentsEnv-v0', render_mode=None, string_mode="training")
q_1, q_2, q_3 = q_training(env, max_epochs=20000, alpha = 0.001, gamma=0.1, min_epsilon=0.1, print_process=True)

q_1_df = pd.DataFrame(q_1, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])    
q_2_df = pd.DataFrame(q_2, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])    
q_3_df = pd.DataFrame(q_3, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])  
  
q_1_df.to_csv(f"{FOLDER_NAME}/three_agents_q1.csv", index=False)
q_2_df.to_csv(f"{FOLDER_NAME}/three_agents_q2.csv", index=False)
q_3_df.to_csv(f"{FOLDER_NAME}/three_agents_q3.csv", index=False)