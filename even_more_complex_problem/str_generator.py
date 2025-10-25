import random

class StringGenerator:
    
    def __init__(self, max_star=5):
        self.max_star = max_star

    def generate_training_str(self):        
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
        choice = random.randint(1, 5)
        
        if choice == 1:
            # Pattern: axac
            return "axac"
        
        elif choice == 2:
            # Pattern: aaxac
            return "aaxac"
        
        elif choice == 3:
            # Pattern: aaaaa*xac
            # This means: "a" followed by 4 or more "a"s, then "xac"
            num_a = random.randint(4, 4+self.max_star)  # Generate 4-10 'a's (4 is minimum)
            return "a" + "a" * num_a + "xac"
        
        elif choice == 4:
            # Pattern: aaxc
            return "aaxc"
        
        else:  # choice == 5
            # Pattern: a(xsa+ayta+aazra)(xsa+ayta+aazra)*axc
            # This means: "a" followed by one of (xsa, ayta, aazra), 
            # then zero or more repetitions of (xsa, ayta, aazra), then "axc"
            
            # First required group
            first_group = random.choice(["xsa", "ayta", "aazra"])
            
            # Zero or more additional groups
            num_additional = random.randint(0, self.max_star)  # 0 to 3 additional groups
            additional_groups = ""
            
            for _ in range(num_additional):
                additional_groups += random.choice(["xsa", "ayta", "aazra"])
            
            return "a" + first_group + additional_groups + "axc"

    def generate_simulation_str(self):
        pass
