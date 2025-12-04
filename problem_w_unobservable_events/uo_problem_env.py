import gymnasium as gym
import numpy as np
from IPython.display import clear_output
import time
from str_generator import StringGenerator
import pandas as pd

gym.register(
    id="UOEnv-v0",
    entry_point="uo_problem_env:UOEnv",
)

class UOEnv(gym.Env):
    COMMUNICATE_COST = 15
    
    # symbol replacement
    #    a1 -> a,   c1 -> c
    #   b21 -> x,  b22 -> y,  b23 -> z
    #   e21 -> s,  e22 -> t,  e23 -> r
    
    # Actual transitions of the system
    simulation_transitions={
        1: {'a':2, 'd':3},
        2: {'d':4, 'x':6},
        3: {'a':5},
        4: {'a':2},
        5: {'g':12, 'a':8, 'd':20, 'f':17}, 
        6: {'a':7},
        7: {'c':9},
        8: {'x':10},
        10:{'c':11},
        12:{'a':13},
        13:{'a':14},
        14:{'z':15},
        15:{'r':16},
        16:{'a':5},
        17:{'a':18},
        18:{'y':19},
        19:{'t':16},
        20:{'x':21},
        21:{'s':16}  
    }
    
    # Converting agent beliefs to states in agent 0 observer for better readability
    m_bottom={
        1:  {1,3},
        2:  {2,4,5,12,20,17},
        3:  {6,21},
        4:  {6,10},
        5:  {2,4,8,13,18},
        6:  {6},
        7:  {7},
        8:  {2,4,14},
        9:  {9},
        10: {10},
        11: {11},
        12: {2,4},
        13: {5,12,17,20},
        14: {14},
        15: {15},
        16: {16},
        17: {8,13,18},
        19: {19},
        21: {21},
        -1: {-1},
    }
    
    agent_0_transitions={
        1:{'a':2},
        2:{'x':3, 'a':5},
        3:{'a':7, 's':16},
        4:{'a':7, 'c':11},
        5:{'a':8, 'y':19, 'x':4},
        6:{'a':7},
        7:{'c':9},
        8:{'a':12, 'z':15, 'x':6},
        10:{'c':11},
        12:{'a':12, 'x':6},
        13:{'a':17, 'x':21},
        14:{'z':15},
        15:{'r':16},
        16:{'a':13},
        17:{'x':10, 'y':19, 'a':14,},
        19:{'t':16},
        21:{'s':16},
    }

    bottom_trantitions={
        1:{'a':2,                  'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        2:{'x':3, 'a':5,           'c':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        3:{'a':7, 's':16,          'c':-1, 'x':-1, 'y':-1, 'z':-1, 't':-1, 'r':-1},
        4:{'a':7, 'c':11,          'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        5:{'a':8, 'y':19, 'x':4,   'c':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        6:{'a':7,                  'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        7:{'c':9,                  'a':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        8:{'a':12, 'z':15,'x':6,   'c':-1, 'y':-1, 's':-1, 't':-1, 'r':-1},
        10:{'c':11,                'a':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        12:{'a':12, 'x':6,         'c':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        13:{'a':17, 'x':21,        'c':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        14:{'z':15,                'a':-1, 'c':-1, 'x':-1, 'y':-1, 's':-1, 't':-1, 'r':-1},
        15:{'r':16,                'a':-1, 'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1},
        16:{'a':13,                'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        17:{'x':10, 'y':19,'a':14, 'c':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        19:{'t':16,                'a':-1, 'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 'r':-1},
        21:{'s':16,                'a':-1, 'c':-1, 'x':-1, 'y':-1, 'z':-1, 't':-1, 'r':-1},
        -1:{'a':-1, 'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1}
    }
    
    metadata = {'render_modes': ['human'], 'string_modes': ['training', 'simulation', 'stats']}
    
    def __init__(self, render_mode=None, string_mode="training", max_star=5):
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: communicate, 1:don't communicate
        self.observation_space = gym.spaces.Box(low=-1, high=21, shape=(3,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.metadata['render_modes']
        assert string_mode is None or string_mode in self.metadata['string_modes']
        self.render_mode = render_mode
        self.string_mode = string_mode
        
        self.string_generator = StringGenerator(max_star=max_star)
        
        if self.string_mode == 'stats':
            df = pd.read_csv("problem_w_unobservable_events/strings.csv")
            self.string_list = df["strings"].to_list()
            self.index = 0
        
    def reset(self, seed=None, options=None):
        # Todo: Generate string
        
        self.string_index = 0
        
        self.agent_0_state = 1
        
        # Initialize agents' partial observations 
        self.agent_1_belief = 1
        self.agent_2_belief = 1
        
        self.communication_count = 0
        
        if self.string_mode == "simulation":
            self.string=self.string_generator.generate_simulation_str()+"$"
            if self.render_mode == 'human':
                print(f"\nSimulation String: {self.string}")
                self.simulate()
            if self.string[self.string_index] == 'd':
                self.agent_0_state = self.simulation_transitions[self.agent_0_state].get('d')
                self.string_index += 1
            if self.render_mode == 'human':
                self.simulate()
        elif self.string_mode == "training":
            self.string=self.string_generator.generate_training_str()+"$" 
            # self.string = "aaazraaazraaazraaazraaxc$"       
            if self.render_mode == 'human':
                print(f"\nNew Episode: {self.string}")
                self.render()
        elif self.string_mode == "stats":
            self.string=self.string_list[self.index]
            self.index = (self.index + 1) % len(self.string_list)
            self.string += "$"
            
        
        config = (self.agent_0_state, self.agent_1_belief, self.agent_2_belief)
        
        info = {"input_alphabet":self.string[self.string_index], "string":self.string }
        
        return np.array(config, dtype=np.int32), info
    
    def step(self, action):
        if self.string_mode == "simulation":
            return self.simulation_step(action)
        
        reward = 0
        agent_id, communicate = action
        terminated = False
        
        curr_symbol=self.string[self.string_index]
        
        self.agent_0_state = self.agent_0_transitions[self.agent_0_state].get(curr_symbol)
        if agent_id==1:
            self.agent_1_belief = self.bottom_trantitions[self.agent_1_belief].get(curr_symbol)
            if communicate == 1:
                reward-=self.COMMUNICATE_COST
                self.agent_2_belief = self.bottom_trantitions[self.agent_2_belief].get(curr_symbol)
        elif agent_id==2:
            self.agent_2_belief = self.bottom_trantitions[self.agent_2_belief].get(curr_symbol)
            if communicate == 1:
                reward-=self.COMMUNICATE_COST
                self.agent_1_belief = self.bottom_trantitions[self.agent_1_belief].get(curr_symbol)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
                
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate==1 else 'did not communicate'} on '{curr_symbol}'")
            self.render()
        
        self.string_index += 1
        
        curr_symbol=self.string[self.string_index]
        
        # reward assignment
        if (self.agent_1_belief != self.agent_0_state) and (self.agent_2_belief != self.agent_0_state):
            reward -= 100
        
        if self.agent_1_belief == -1 and self.agent_2_belief == -1:
            reward -=500
            terminated = True
        elif self.agent_0_state == 11 and not(self.agent_1_belief == 11 or self.agent_2_belief == 11):
            # Penalized configuration condition 1-1
            reward -=500
            terminated = True
        elif self.agent_0_state == 9 and not(self.agent_1_belief == 9 or self.agent_2_belief == 9):
            # Penalized configuration condition 1-2
            reward -=500
            terminated = True
        elif self.agent_0_state != 11 and self.agent_1_belief in [11,-1] and self.agent_2_belief in [11,-1]:
            # Penalized configuration condition 2-1
            reward -=500
            terminated = True
        elif self.agent_0_state != 9 and self.agent_1_belief in [9,-1] and self.agent_2_belief in [9,-1]:
            # Penalized configuration condition 2-2
            reward -=500
            terminated = True
        elif self.string[self.string_index]=="$":
            terminated = True
        
        
        config = (self.agent_0_state, self.agent_1_belief, self.agent_2_belief)
        info = {"input_alphabet":self.string[self.string_index], "string":self.string}
        return np.array(config, dtype=np.int32), reward, terminated, False, info
    
    def render(self):
        print(f"Current symbol: '{self.string[self.string_index]}'")
        # print(self.agent_0_state, self.m_bottom[self.agent_0_state])
        # print(self.agent_1_belief, self.agent_2_belief)
        # print(self.m_bottom[self.agent_1_belief], self.m_bottom[self.agent_2_belief])
        print(f"Config: <{self.agent_0_state}:{self.m_bottom[self.agent_0_state]}, {self.agent_1_belief}:{self.m_bottom[self.agent_1_belief]}, {self.agent_2_belief}:{self.m_bottom[self.agent_2_belief]}>")
    
    def simulation_step(self, action):
        # Note: In simulation mode, we still use variable "agent_0_state", but this refers to the actual global state.
        agent_id, communicate = action
        terminated = False
        
        curr_symbol=self.string[self.string_index]
        
        if curr_symbol == 'c':
            if self.agent_0_state not in [7,10]:
                raise ValueError("Disable action can only be taken at state 7 or 10 in simulation.")
            
            # Control Policy: If in state 7:{7}, disable 'c'
            agent_1_disable_c = self.agent_1_belief == 7
            agent_2_disable_c = self.agent_2_belief == 7
            
            
            if not (agent_1_disable_c or agent_2_disable_c):
                self.agent_0_state = self.simulation_transitions[self.agent_0_state].get(curr_symbol)
                self.agent_1_belief = self.bottom_trantitions[self.agent_1_belief].get(curr_symbol)
                if communicate == 1:
                    self.communication_count+=1
                    self.agent_2_belief = self.bottom_trantitions[self.agent_2_belief].get(curr_symbol)
            
            if self.render_mode == 'human':
                self.simulate(True, agent_1_disable_c, agent_2_disable_c)
        else:
            self.agent_0_state = self.simulation_transitions[self.agent_0_state].get(curr_symbol)
            if agent_id==1:
                self.agent_1_belief = self.bottom_trantitions[self.agent_1_belief].get(curr_symbol)
                if communicate ==1:
                    self.communication_count+=1
                    self.agent_2_belief = self.bottom_trantitions[self.agent_2_belief].get(curr_symbol)
            elif agent_id==2:
                self.agent_2_belief = self.bottom_trantitions[self.agent_2_belief].get(curr_symbol)
                if communicate == 1:
                    self.communication_count+=1
                    self.agent_1_belief = self.bottom_trantitions[self.agent_1_belief].get(curr_symbol)
            else:
                raise ValueError("Invalid agent_id. Must be 1 or 2.")
            
            if self.render_mode == 'human' and communicate == 1:
                print(f"\nAgent {agent_id} communicated on '{curr_symbol}'")

            if self.render_mode == 'human':
                self.simulate()
        
        self.string_index += 1
        
        curr_symbol=self.string[self.string_index]
        
        if curr_symbol in ['d','g', 'f']:
            self.agent_0_state = self.simulation_transitions[self.agent_0_state].get(curr_symbol)
            if self.render_mode == 'human':
                self.simulate()
            
            self.string_index += 1
            curr_symbol=self.string[self.string_index]
        
        
        if self.string[self.string_index]=="$":
            terminated = True
        
        config = (self.agent_0_state, self.agent_1_belief, self.agent_2_belief)
        info = {"input_alphabet":self.string[self.string_index], "string":self.string}
        
        return np.array(config, dtype=np.int32), -1, terminated, False, info
    
    def simulate(self, seven_or_ten=False, agent_1_disable_c=None, agent_2_disable_c=None):
        # Note: In simulation mode, we still use variable "agent_0_state", but this refers to the actual global state.
        
        a = [" " for _ in range(22)]
        a[self.agent_0_state] = "#"
        
        print(f"global state: {self.agent_0_state},\n agent 1's belief: {self.agent_1_belief}:{self.m_bottom[self.agent_1_belief]},\n agent 2's belief: {self.agent_2_belief}:{self.m_bottom[self.agent_2_belief]}>,\n Current symbol: '{self.string[self.string_index]}', # comm: {self.communication_count}")
        
        
        block = "|"
        
        if seven_or_ten:
            if agent_1_disable_c or agent_2_disable_c:
                block = "X" 
            if agent_1_disable_c:
                print(f"Agent 1 disables 'c' at state {self.agent_1_belief}")
            if agent_2_disable_c:
                print(f"Agent 2 disables 'c' at state {self.agent_2_belief}")
            
            if self.agent_0_state == 7 and not (agent_1_disable_c or agent_2_disable_c):
                print("Failed to disable 'c'")
            elif self.agent_0_state == 10 and (agent_1_disable_c or agent_2_disable_c):
                print("Failed to enable 'c'")
            else:
                print("Successfully controlled 'c'")
            
        
        
        print(
    "           start             \n"
    "             |               \n"
    "             V               \n"
    "           +-1-+             \n"
   f"           | {a[1]} |             \n"
    "           +---+             \n"
    "           /   \             \n"
    "        a /     \ d          \n"
    "         V       V           \n"
    "       +-2-+     +-3-+        \n"
   f"   --->| {a[2]} |     | {a[3]} |-------- \n"
    "   |   +---+     +---+        |    \n" 
    " a |   /   |                  | a  \n"
    "   |  / d  | x                |    \n"
    "   | V     V                  V \n"
    "  +-4-+  +-6-+              +-5-+\n"
   f"  | {a[4]} |  | {a[6]} |       ------>| {a[5]} |----------\n"
    "  +---+  +---+      /       +---+          \       \n"
    "         /         /       /  |   \         \      \n"
    "      a /         /     g /   | a  \ d       \ f   \n"
    "       V         /       V    V     V         V    \n"
    "   +-7-+        /   +-12-+  +-8-+   +-20-+  +-17-+ \n"
   f"   | {a[7]} |       |    | {a[12]}  |  | {a[8]} |   | {a[20]}  |  | {a[17]}  | \n"
    "   +---+       |    +----+  +---+   +----+  +----+ \n"
    f"     {block}         |      |       {block}       |       |    \n"
    f"   c {block}         |    a |     x {block}       |       | a  \n"
    "     V         |      V       V       |       V    \n"
    "   +=9=+       |    +-13-+  +-10-+    |     +-18-+ \n"
   f"  || {a[9]} ||      |    | {a[13]}  |  | {a[10]}  |    | x   | {a[18]}  | \n"
    "   +===+       |    +----+  +----+    |     +----+ \n"
    "               |      |       |       |       |    \n"
    "               |    a |     c |       |       | y  \n"
    "               |      V       V       V       V    \n"
    "               |    +-14-+  +=11=+  +-21-+  +-19-+ \n"
   f"             a |    | {a[14]}  |  ||{a[11]} ||  | {a[21]}  |  | {a[19]}  | \n"
    "               |    +----+  +====+  +----+  +----+ \n"
    "               |      |               |       |    \n"
    "               |    z |               |       |    \n"
    "               |      V               | s     |    \n"
    "               |    +-15-+            |       | t  \n"
   f"               |    | {a[15]}  |            |       |    \n"
    "                \   +----+            V       |    \n"
    "                 \      \    r     +-16-+     |    \n"
   f"                  \      --------->| {a[16]}  |<-----    \n"
    "                   \               +----+          \n"
    "                    \_________________|            \n") 
        time.sleep(1)
        clear_output()
        print()