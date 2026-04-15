import random
import pandas as pd

class WordGenerator:
    def __init__(self, max_star=5,):
        self.max_star = max_star
    
    def generate_training_word(self):
        word = ''
        
        word += 's' * random.randint(0, self.max_star)
        
        n = random.choice(['a', 'b', 'c'])
        word += n
        if n == 'a':
            word += 's' * random.randint(0, self.max_star)
            n = random.choice(['p', 'm'])
            word += n
            if n == 'p':
                word += 's' * random.randint(0, self.max_star)
                word += 'ms'
            else:
                word += 's' * random.randint(0, self.max_star)
                word += random.choice(['ps', 'ms'])    
        
        elif n == 'b':
            word += 's' * random.randint(0, self.max_star)
            n = random.choice(['q', 'r'])
            word += n
            word += 's' * random.randint(0, self.max_star)
            word += 'ms'
        
        else:
            word += 's' * random.randint(0, self.max_star)
            n = random.choice(['p', 'q'])
            word += n
            word += 's' * random.randint(0, self.max_star)
            word += 'ms'
        
        return word

# generator = WordGenerator(max_star=2)

# testing_pool = []

# count = 0

# while count<100:
#     word = generator.generate_training_word()
#     if word not in testing_pool:
#         testing_pool.append(word)
#         count+=1

# testing_pool_df = pd.DataFrame(testing_pool, columns=['word'])

# testing_pool_df.to_csv('ex_4_check_inference_observability/simulation_words.csv', index=False)        
        
