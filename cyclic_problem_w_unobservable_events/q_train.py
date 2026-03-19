import gymnasium as gym
import pandas as pd
import cyclic_problem_env
from cyclic_problem_q import q_training
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

q_training_env = gym.make('CyclicEnv-v0', render_mode=None, string_mode="training")

q_1, q_2 = q_training(q_training_env, epochs=10000, alpha=0.01, gamma=0.1, epsilon=0.1, print_process=True)

q_1_df = pd.DataFrame(q_1, columns=["do not communcate", "communicate"])
q_2_df = pd.DataFrame(q_2, columns=["do not communcate", "communicate"])

q_1_df.to_csv(f'cyclic_problem_w_unobservable_events/demo_q1_table.csv')
q_2_df.to_csv(f'cyclic_problem_w_unobservable_events/demo_q2_table.csv')