
import numpy as np
import random

class WordGenerator:
    def __init__(self, max_star=5):
        self.max_star = max_star

    def generate_simulation_word(self):

        return random.choice(['cax', 'cay', 'acx', 'acy']) + 's'
    
    def generate_training_word(self):
        return random.choice(['cax', 'cay', 'acx', 'acy']) + 's'
    
    # def generate_simulation_word(self):
        
    #     word = ""

    #     choice =  random.choice([0,1])
        
    #     if choice == 0:
    #         word += 'ac'
    #         word += random.choice(['x', 'y'])
    #     else:
    #         word += 'd'
    #         choice = random.choice([0,1])
    #         if choice == 0:
    #             word += 'ca'
    #             word += random.choice(['x','y'])
    #         else:
    #             word += 'd'
    #             word += random.choice(['xy', 'yx'])
        
    #     return word + 's'
    
    # def generate_training_word(self):
    #     word = ""

    #     choice =  random.choice([0,1, 2])
        
    #     if choice == 0:
    #         word += random.choice(['cax', 'cay'])
    #     elif choice == 1:
    #         word += random.choice(['acx', 'acy'])
    #     else:
    #         word += random.choice(['xy', 'yx'])
        
    #     return word + 's'

import pandas as pd
generator = WordGenerator(max_star=5)

testing_pool = []

count = 0

while count<1000:
    word = generator.generate_simulation_word()
    if word not in testing_pool:
        testing_pool.append(word)
    count+=1

testing_pool_df = pd.DataFrame(testing_pool, columns=['word'])

testing_pool_df.to_csv('three_agents_exp/words_for_stats.csv', index=False)