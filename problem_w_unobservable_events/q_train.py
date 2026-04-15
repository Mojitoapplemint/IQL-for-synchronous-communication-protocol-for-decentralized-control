import gymnasium as gym
import pandas as pd
import uo_problem_env
from uo_problem_q import q_training

q_training_env = gym.make('UOEnv-v0', render_mode=None, string_mode="training")

q_1, q_2 = q_training(q_training_env, epochs=10000, alpha=0.01, gamma=0.9, epsilon=0.1)

q_1_df = pd.DataFrame(q_1, columns=["do not communcate", "communicate"])
q_2_df = pd.DataFrame(q_2, columns=["do not communcate", "communicate"])

q_1_df.to_csv(f'problem_w_unobservable_events/demo_q1_table.csv')
q_2_df.to_csv(f'problem_w_unobservable_events/demo_q2_table.csv')
