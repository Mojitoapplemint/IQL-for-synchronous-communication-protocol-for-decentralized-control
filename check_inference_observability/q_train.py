import gymnasium as gym
import pandas as pd
import inference_env
from inference_q import q_training, FOLDER_NAME
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

q_training_env = gym.make('InferenceEnv-v0', render_mode=None, string_mode="training")

q_1, q_2 = q_training(q_training_env, epochs=1000000, alpha=0.01, gamma=0.2, epsilon=0.1, print_process=True)

q_1_df = pd.DataFrame(q_1, columns=["do not communicate", "communicate"])
q_2_df = pd.DataFrame(q_2, columns=["do not communicate", "communicate"])

q_1_df.to_csv(f'{FOLDER_NAME}/q_1.csv')
q_2_df.to_csv(f'{FOLDER_NAME}/q_2.csv')