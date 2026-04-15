import gymnasium as gym
import numpy as np
from IPython.display import clear_output
import time
from word_generator import WordGenerator
import pandas as pd

gym.register(
    id="UOEnv-v1",
    entry_point="uo_problem_env:UOEnv",
)

class UOEnv(gym.Env):
    COMMUNICATE_COST = 1
    PENALTY_E = 50 # For Exp 1
    # PENALTY_E = 25 # For Exp 2
    # PENALTY_E = 0   # For Exp 3
    PENALTY_D =50
    
    # 45,1000,1000
    
    D_PEN_STATES = {9}
    
    E_PEN_STATES = {11}
    
    STATES_DISABLE_SIGMA = {7}
    
    # symbol replacement
    #    a1 -> a,   c1 -> c
    #   b21 -> x,  b22 -> y,  b23 -> z
    #   e21 -> s,  e22 -> t,  e23 -> r
    
    # Actual transitions of the system
    m_L_transitions={
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
    
    # Converting agent beliefs to states in observer for better readability
    m_L_bot={
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
    
    observer_sigma_o={
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

    m_L_bot_transition={
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
    
    metadata = {'render_modes': ['human'], 'string_modes': ['training', 'simulation']}
    
    def __init__(self, render_mode=None, string_mode="training", max_star=5):
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: communicate, 1:don't communicate
        self.observation_space = gym.spaces.Box(low=-1, high=21, shape=(3,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.metadata['render_modes']
        assert string_mode is None or string_mode in self.metadata['string_modes']
        self.render_mode = render_mode
        self.string_mode = string_mode
        
        self.string_generator = WordGenerator(max_star=max_star)
        
        df = pd.read_csv("problem_w_unobservable_events/simulation_words.csv")
        self.simulation_word_list = df["word"].to_list()
        self.simulation_word_index = 0
        
    def reset(self, seed=None, options=None):
        
        self.event_index = 0
        
        self.system_state = 1
        
        # Initialize agents' partial observations 
        self.agent_1_belief = 1
        self.agent_2_belief = 1
        
        self.communication_count = 0
        
        if self.string_mode == "simulation":
            self.word=self.simulation_word_list[self.simulation_word_index]
            self.simulation_word_index = (self.simulation_word_index + 1) % len(self.simulation_word_list)
            self.word += "$"
            if self.render_mode == 'human':
                print(f"\nSimulation String: {self.word}")
                self.simulate()
            if self.word[self.event_index] == 'd':
                self.system_state = self.m_L_transitions[self.system_state].get('d')
                self.event_index += 1
            if self.render_mode == 'human':
                self.simulate()
        elif self.string_mode == "training":
            self.word=self.string_generator.generate_training_word()+"$" 
            # self.string = "aaazraaazraaazraaazraaxc$"       
            if self.render_mode == 'human':
                print(f"\nNew Episode: {self.word}")
                self.render()
        
        config = (self.system_state, self.agent_1_belief, self.agent_2_belief)
        
        info = {"curr_event":self.word[self.event_index], "word":self.word }
        
        return np.array(config, dtype=np.int32), info
    
    def step(self, action):
        if self.string_mode == "simulation":
            return self.simulation_step(action)
        
        comm_cost = 0
        penalty = 0
        agent_id, communicate = action
        terminated = False
        
        curr_symbol=self.word[self.event_index]
        
        self.system_state = self.observer_sigma_o[self.system_state].get(curr_symbol)
        if agent_id==1:
            self.agent_1_belief = self.m_L_bot_transition[self.agent_1_belief].get(curr_symbol)
            if communicate == 1:
                comm_cost-=self.COMMUNICATE_COST
                self.agent_2_belief = self.m_L_bot_transition[self.agent_2_belief].get(curr_symbol)
        elif agent_id==2:
            self.agent_2_belief = self.m_L_bot_transition[self.agent_2_belief].get(curr_symbol)
            if communicate == 1:
                comm_cost-=self.COMMUNICATE_COST
                self.agent_1_belief = self.m_L_bot_transition[self.agent_1_belief].get(curr_symbol)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
                
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate==1 else 'did not communicate'} on '{curr_symbol}'")
            self.render()
        
        self.event_index += 1
        
        curr_symbol=self.word[self.event_index]
        
        # reward assignment        
        # if (self.agent_1_belief != self.agent_0_state) and (self.agent_2_belief != self.agent_0_state):
        #     reward -= 15


        if self.system_state in self.E_PEN_STATES and  self.agent_1_belief ==-1 and self.agent_2_belief ==-1:
            
            penalty -=self.PENALTY_E
            terminated = True

        if self.system_state in self.D_PEN_STATES and  self.agent_1_belief ==-1 and self.agent_2_belief ==-1:
            
            penalty -=self.PENALTY_D
            terminated = True
        elif self.word[self.event_index]=="$":
            terminated = True
        

        
        config = (self.system_state, self.agent_1_belief, self.agent_2_belief)
        info = {"curr_event":self.word[self.event_index], "word":self.word}
        return np.array(config, dtype=np.int32), (comm_cost,penalty), terminated, False, info
    
    def render(self):
        print(f"Current symbol: '{self.word[self.event_index]}'")
        # print(self.agent_0_state, self.m_bottom[self.agent_0_state])
        # print(self.agent_1_belief, self.agent_2_belief)
        # print(self.m_bottom[self.agent_1_belief], self.m_bottom[self.agent_2_belief])
        print(f"Config: <{self.system_state}:{self.m_L_bot[self.system_state]}, {self.agent_1_belief}:{self.m_L_bot[self.agent_1_belief]}, {self.agent_2_belief}:{self.m_L_bot[self.agent_2_belief]}>")
    
    def simulation_step(self, action):
        # Note: In simulation mode, we still use variable "agent_0_state", but this refers to the actual global state.
        agent_id, communicate = action
        terminated = False
        comm_cost = 0
        penalty = 0
        simulation_result = False
        
        curr_symbol=self.word[self.event_index]
        
        if curr_symbol == 'c':
            if self.system_state not in [7,10]:
                raise ValueError("Disable action can only be taken at state 7 or 10 in simulation.")
            
            # Control Policy: If in state 7:{7}, disable 'c'
            agent_1_disable_c = self.agent_1_belief == 7
            agent_2_disable_c = self.agent_2_belief == 7
            
            
            if not (agent_1_disable_c or agent_2_disable_c):
                self.system_state = self.m_L_transitions[self.system_state].get(curr_symbol)
                self.agent_1_belief = self.m_L_bot_transition[self.agent_1_belief].get(curr_symbol)
                if communicate == 1:
                    self.communication_count+=1
                    comm_cost-=self.COMMUNICATE_COST
                    self.agent_2_belief = self.m_L_bot_transition[self.agent_2_belief].get(curr_symbol)
            
            if self.render_mode == 'human':
                self.simulate(True, agent_1_disable_c, agent_2_disable_c)
                
            if (self.system_state in self.E_PEN_STATES) and not (agent_1_disable_c or agent_2_disable_c):
                simulation_result = True

            if (self.system_state in self.STATES_DISABLE_SIGMA) and (agent_1_disable_c or agent_2_disable_c):
                simulation_result = True
                
        else:
            self.system_state = self.m_L_transitions[self.system_state].get(curr_symbol)
            if agent_id==1:
                self.agent_1_belief = self.m_L_bot_transition[self.agent_1_belief].get(curr_symbol)
                if communicate ==1:
                    self.communication_count+=1
                    comm_cost-=self.COMMUNICATE_COST
                    self.agent_2_belief = self.m_L_bot_transition[self.agent_2_belief].get(curr_symbol)
            elif agent_id==2:
                self.agent_2_belief = self.m_L_bot_transition[self.agent_2_belief].get(curr_symbol)
                if communicate == 1:
                    self.communication_count+=1
                    comm_cost-=self.COMMUNICATE_COST
                    self.agent_1_belief = self.m_L_bot_transition[self.agent_1_belief].get(curr_symbol)
            else:
                raise ValueError("Invalid agent_id. Must be 1 or 2.")
            
            if self.render_mode == 'human' and communicate == 1:
                print(f"\nAgent {agent_id} communicated on '{curr_symbol}'")

            if self.render_mode == 'human':
                print(f"\nAgent {agent_id} did not communicate on '{curr_symbol}'")
                self.simulate()
    
        
        
        
        self.event_index += 1
        
        curr_symbol=self.word[self.event_index]
        
        if curr_symbol in ['d','g', 'f']:
            self.system_state = self.m_L_transitions[self.system_state].get(curr_symbol)
            if self.render_mode == 'human':
                self.simulate()
            
            self.event_index += 1
            curr_symbol=self.word[self.event_index]
        
        
        if self.system_state in self.E_PEN_STATES and  self.agent_1_belief ==-1 and self.agent_2_belief ==-1:
            
            penalty -=self.PENALTY_E
            terminated = True

        if self.system_state in self.D_PEN_STATES and  self.agent_1_belief ==-1 and self.agent_2_belief ==-1:
            
            penalty -=self.PENALTY_D
            terminated = True
            
        elif self.word[self.event_index]=="$":
            terminated = True
        
        if self.word[self.event_index]=="$":
            terminated = True
        
        config = (self.system_state, self.agent_1_belief, self.agent_2_belief)
        info = {"curr_event":self.word[self.event_index], "word":self.word}
        
        return np.array(config, dtype=np.int32), (comm_cost, penalty), terminated, simulation_result, info
    
    def simulate(self, seven_or_ten=False, agent_1_disable_c=None, agent_2_disable_c=None):
        # Note: In simulation mode, we still use variable "agent_0_state", but this refers to the actual global state.
        
        a = [" " for _ in range(22)]
        a[self.system_state] = "#"
        
        print(f"global state: {self.system_state},\n agent 1's belief: {self.agent_1_belief}:{self.m_L_bot[self.agent_1_belief]},\n agent 2's belief: {self.agent_2_belief}:{self.m_L_bot[self.agent_2_belief]}>,\n Current symbol: '{self.word[self.event_index]}', # comm: {self.communication_count}")
        
        
        block = "|"
        
        if seven_or_ten:
            if agent_1_disable_c or agent_2_disable_c:
                block = "X" 
            if agent_1_disable_c:
                print(f"Agent 1 disables 'c' at state {self.agent_1_belief}")
            if agent_2_disable_c:
                print(f"Agent 2 disables 'c' at state {self.agent_2_belief}")
            
            if self.system_state == 7 and not (agent_1_disable_c or agent_2_disable_c):
                print("Failed to disable 'c'")
            elif self.system_state == 10 and (agent_1_disable_c or agent_2_disable_c):
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
    "           /   \\             \n"
    "        a /     \\ d          \n"
    "         V       V           \n"
    "       +-2-+     +-3-+        \n"
   f"   --->| {a[2]} |     | {a[3]} |-------- \n"
    "   |   +---+     +---+        |    \n" 
    " a |   /   |                  | a  \n"
    "   |  / d  | x                |    \n"
    "   | V     V                  V \n"
    "  +-4-+  +-6-+              +-5-+\n"
   f"  | {a[4]} |  | {a[6]} |       ------>| {a[5]} |----------\n"
    "  +---+  +---+      /       +---+          \\       \n"
    "         /         /       /  |   \\         \\      \n"
    "      a /         /     g /   | a  \\ d       \\ f   \n"
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
    "                \\   +----+            V       |    \n"
    "                 \\      \\    r     +-16-+     |    \n"
   f"                  \\      --------->| {a[16]}  |<-----    \n"
    "                   \\               +----+          \n"
    "                    \\_________________|            \n") 
        time.sleep(1)
        clear_output()
        print()