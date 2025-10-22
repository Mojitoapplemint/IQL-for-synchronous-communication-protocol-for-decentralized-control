import random

'''
This code is written by AI and modified by a human
'''

class RegexStringGenerator:
    def __init__(self, max_star=10,):
        self.max_star = max_star
    
    def set_max_star(self, max_star):
        self.max_star = max_star
        
    
    def generate_ay_star_x(self):
        """Generate pattern: ay*x (a followed by zero or more y's followed by x)"""
        result = "a"
        # Add zero or more y's
        y_count = random.randint(0, self.max_star)
        result += "y" * y_count
        result += "x"
        return result
    
    def generate_by_star_b(self):
        """Generate pattern: by*b (b followed by zero or more y's followed by b)"""
        result = "b"
        # Add zero or more y's
        y_count = random.randint(0, self.max_star)
        result += "y" * y_count
        result += "b"
        return result
    
    def generate_by_star_xa(self):
        """Generate pattern: by*xa (b followed by zero or more y's followed by xa)"""
        result = "b"
        # Add zero or more y's
        y_count = random.randint(0, self.max_star)
        result += "y" * y_count
        result += "xa"
        return result
    
    def gen_x_plus_y_star(self):
        """Generate (x+y)* """
        result = ""
        length = random.randint(0, self.max_star)
        for _ in range(length):
            result += random.choice(["x", "y"])
        return result
    
    def gen_y_star(self):
        """Generate y* """
        result = ""
        length = random.randint(0, self.max_star)
        for _ in range(length):
            result += "y"
        return result
    
    def generate_complex_pattern(self):
        """Generate the complex pattern: ((b+ay*x)a+ay*b(by*b)*(a+by*xa))"""
        choice = random.choice([1, 2])
        
        if choice == 1:
            # Pattern: (b+ay*x)a
            sub_choice = random.choice([1, 2])
            if sub_choice == 1:
                return "ba"
            else:
                return self.generate_ay_star_x() + "a"
        else:
            # Pattern: ay*b(by*b)*(a+by*xa)
            result = "a"
            # Add zero or more y's before b
            y_count1 = random.randint(0, 2)
            result += "y" * y_count1
            result += "b"
            
            # Add (by*b)* zero or more times
            repeat_count = random.randint(0, 2)
            for _ in range(repeat_count):
                result += self.generate_by_star_b()
            
            # Add (a+by*xa)
            sub_choice = random.choice([1, 2])
            if sub_choice == 1:
                result += "a"
            else:
                result += self.generate_by_star_xa()
            
            return result
    
    def generate_main_pattern(self):
        """Generate the main repeating pattern"""
        result = ""
        
        # Determine length of the repeating part
        repeat_length = random.randint(0, self.max_star)
        
        for _ in range(repeat_length):
            choice = random.choice([1, 2, 3])
            if choice == 1:
                result += "x"
            elif choice == 2:
                result += "y"
            else:
                # Add the complex pattern followed by y
                result += self.generate_complex_pattern() + "y"
        
        return result
    
    def generate_half_training_str(self):
        # Build the complete string
        result = ""
        
        # Add the repeating part: (x+y+((b+ay*x)a+ay*b(by*b)*(a+by*xa))y)*
        result += self.generate_main_pattern()
        
        # Add the final part: ((b+ay*x)a+ay*b(by*b)*(a+by*xa))x
        result += self.generate_complex_pattern() + "x"
        
        return result
    
    def generate_maintenance_str(self):
        result=""
        
        total_count = random.randint(0, self.max_star)
        
        for _ in range(total_count):
            choice = random.choice([0, 1])
            if choice == 0:
                # First part: (y(x+y)*b + y(x+y)*a y* x) a
                subchoice = random.choice([0, 1])
                if subchoice == 0:
                    # y(x+y)*b a
                    result += "y" + self.gen_x_plus_y_star() + "b" + "a"
                else:
                    # y(x+y)*a y* x a
                    result += "y" + self.gen_x_plus_y_star() + "a" + self.gen_y_star() + "x" + "a"
            else:
                # Second part: y(x+y)*a y* b (b y* b)* (a + b y* x a)
                result += "y" + self.gen_x_plus_y_star() + "a" + self.gen_y_star() + "b"
                # Generate (b y* b)* part
                b_y_b_count = random.randint(0,self.max_star)  # Limit repetitions
                for _ in range(b_y_b_count):
                    result += "b" + self.gen_y_star() + "b"
                
                # Generate (a + b y* x a) part
                subchoice = random.choice([0, 1])
                if subchoice == 0:
                    result += "a"
                else:
                    result += "b" + self.gen_y_star() + "x" + "a"
                    
        return result
    
        
    def generate_full_training_str(self):
        result = self.generate_half_training_str()[:-1]
                
        num_maintenance_str = random.randint(0, self.max_star)
        
        for _ in range(num_maintenance_str):
            result += self.generate_maintenance_str()        
            
        return result + "x"
    
    def generate_simulation_str(self):
        result = self.generate_half_training_str()
                
        num_maintenance_str = random.randint(0, self.max_star)
        
        for _ in range(num_maintenance_str):
            result += self.generate_maintenance_str()
            result += "x"   
            
        return result

