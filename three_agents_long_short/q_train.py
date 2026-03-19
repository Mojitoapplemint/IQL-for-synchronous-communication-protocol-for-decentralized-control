import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_long_short.three_agents_ls_env as three_agents_ls_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_long_short.three_agents_ls_q import FOLDER_NAME, q_training


env = gym.make('ThreeAgentsLSEnv-v0', render_mode=None, string_mode="training")
q_1, q_3 = q_training(env, epochs=1000000, alpha=0.001, gamma=0.99, min_epsilon=0.1, print_process=True)

q_1_df = pd.DataFrame(q_1, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])
q_3_df = pd.DataFrame(q_3, columns=["[X,X]", "[X,O]", "[O,X]", "[O,O]"])

q_1_df.to_csv(f"{FOLDER_NAME}/q1.csv", index=False)
q_3_df.to_csv(f"{FOLDER_NAME}/q3.csv", index=False)