import random

class RegexWordGenerator:
    def __init__(self, max_star=5,):
        self.max_star = max_star
    
    def generate_maintenance_word(self):
        choice = random.choice([0,1])
        n = random.randint(0, self.max_star)
        word = ''
        
        for i in range(n):
            word+='c'
            n2 = random.randint(0, self.max_star)
            word+= 'bac'*n2
            word=='ab'
        
        if choice==0:
            word+='s'
        else:
            word+='c'
            n2 = random.randint(0, self.max_star)
            word+= 'bac'*n2
            word=='bas'
        return word
        
    
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
        choice = random.choice([0,1])
        word = ''
        n = random.randint(1, self.max_star)
        
        for i in range(n):
            choice = random.choice([0,1])
            if choice == 0:
                word += 'abc'
            else:
                word += 'bac'
        
        choice = random.choice([0,1])
        if choice == 0:
            word += 'abs'
        else:
            word += 'bas'
        
        return word
        
        
        
