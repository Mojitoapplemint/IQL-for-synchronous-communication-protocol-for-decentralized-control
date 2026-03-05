import random
import pandas as pd

class WordGenerator:
    def __init__(self, max_star=5,):
        self.max_star = max_star
    
    def generate_training_word(self):
        word = ''
        
        n = random.randint(1, self.max_star)
        for i in range(n):
            choice = random.choice([0,1])
            if choice == 0:
                word += 'ab'
            else:
                word += 'ba'
        return word + 's'
    
    def generate_simulation_word(self):
        word = ''
        
        for _ in range( random.randint(0, self.max_star)):
            
            for _ in range(random.randint(0, self.max_star)):
                word+=random.choice(['abc','bac'])
            
            word += 'absc'
        
        word += random.choice(['abs','bas'])

        
        return word
        

generator = WordGenerator(max_star=6)

testing_pool = []

count = 0

while count<500:
    word = generator.generate_simulation_word()
    if word not in testing_pool:
        testing_pool.append(word)
        count+=1

testing_pool_df = pd.DataFrame(testing_pool, columns=['word'])

testing_pool_df.to_csv('cyclic_problem_w_unobservable_events/simulation_words.csv', index=False)        
        
