import random
import pandas as pd

class WordGenerator:
    
    def __init__(self, max_star=5):
        self.max_star = max_star

    def generate_training_word(self):        
        """
        Generates a string that matches the regex pattern:
        axac+aaxac+aaaaa*xac+aaxc+a(xsa+ayta+aazra)(xsa+ayta+aazra)*axc
        
        The pattern consists of 5 alternatives separated by +:
        1. axac
        2. aaxac  
        3. aaaaa*xac
        4. aaxc
        5. a(xsa+ayta+aazra)(xsa+ayta+aazra)*axc
        """
        
        # Choose one of the 5 alternatives randomly
        choice = random.randint(1, 6)
        
        if choice == 1:
            # Pattern: axac
            return "axac"
        
        elif choice == 2:
            # Pattern: aaxac
            return "aaxac"
        
        elif choice == 3:
            # Pattern: aaaaa*xac
            num_a = random.randint(4, 4+self.max_star)
            return "a" + "a" * num_a + "xac"
        
        elif choice == 4:
            # Pattern: aaxc
            return "aaxc"

        elif choice == 5:
            # Pattern: aaaxac
            return "aaaxac"
        
        else:  # choice == 6
            # Pattern: a(xsa+ayta+aazra)(xsa+ayta+aazra)*axc
            # First required group
            first_group = random.choice(["xsa", "ayta", "aazra"])
            
            # Zero or more additional groups
            num_additional = random.randint(0, self.max_star)
            additional_groups = ""
            
            for _ in range(num_additional):
                additional_groups += random.choice(["xsa", "ayta", "aazra"])
            
            return "a" + first_group + additional_groups + "axc"

    def generate_simulation_word(self): 
        """
        Generates a string that matches the regex pattern:
        a(da)*xac+da(gaazra+dxsa+fayta)*axc
        
        The pattern consists of 2 alternatives separated by +:
        1. a(da)*xac
        2. da(gaazra+dxsa+fayta)*axc
        """
        
        choice = random.randint(1, 2)
        
        if choice == 1: # Pattern: a(da)*xac
            num_da = random.randint(0, self.max_star)  # 0 to 5 repetitions of "da"
            da_repetitions = "da" * num_da
            return "a" + da_repetitions + "xac"
        
        else: # Pattern: da(gaazra+dxsa+fayta)*axc
            num_groups = random.randint(0, self.max_star)  # 0 to 4 repetitions
            middle_groups = ""
            
            for _ in range(num_groups):
                # middle_groups += "fayta"
                middle_groups += random.choice(["gaazra", "dxsa", "fayta"])
            
            return "da" + middle_groups + "axc"


# generator = WordGenerator(max_star=5)

# testing_pool = []

# count = 0

# while count<300:
#     word = generator.generate_simulation_word()
#     if word not in testing_pool:
#         testing_pool.append(word)
#         count+=1

# testing_pool_df = pd.DataFrame(testing_pool, columns=['word'])

# testing_pool_df.to_csv('problem_w_unobservable_events/simulation_words.csv', index=False)