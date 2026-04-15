import random
import pandas as pd

class WordGenerator:
    def __init__(self, max_star=5,):
        self.max_star = max_star
    
    def generate_training_word(self):

        
        return random.choice(['abs', 'adbs', 'bas'])

# generator = WordGenerator(max_star=2)

# testing_pool = []

# count = 0

# while count<100:
#     word = generator.generate_training_word()
#     if word not in testing_pool:
#         testing_pool.append(word)
#     count+=1

# testing_pool_df = pd.DataFrame(testing_pool, columns=['word'])

# testing_pool_df.to_csv('ex_3_check_distributive_observability/simulation_words.csv', index=False)        
        
