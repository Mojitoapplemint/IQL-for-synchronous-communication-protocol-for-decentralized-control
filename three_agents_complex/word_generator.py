
# L_tilde: ((d+b)(a+c)(x+b)b)*(d+b)(a+c)(x+b)s
# L_tilde_projection: ((e+b)(a+c)(x+b)b)*(e+b)(a+c)(x+b)s
import numpy as np
import random

class WordGenerator:
    def __init__(self, max_star=5):
        self.max_star = max_star
    
    def generate_simulation_word(self):
        
        num_iter = random.randint(0, self.max_star)
        
        word = ""
        for _ in range(num_iter):
        
            word += random.choice(['d','b'])
            word += random.choice(['a','c'])
            word += random.choice(['x','b'])
            word += 'b'
        
        word += random.choice(['d','b'])
        word += random.choice(['a','c'])
        word += random.choice(['x','b'])
        word += 's'
        
        return word
    
    def generate_training_word(self):
        num_iter = random.randint(0, self.max_star)
        
        word = ""
        for _ in range(num_iter):
        
            word += random.choice(['','b'])
            word += random.choice(['a','c'])
            word += random.choice(['x','b'])
            word += 'b'
        
        word += random.choice(['','b'])
        word += random.choice(['a','c'])
        word += random.choice(['x','b'])
        word += 's'
        
        return word