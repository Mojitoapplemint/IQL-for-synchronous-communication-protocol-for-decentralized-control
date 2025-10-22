import gymnasium as gym
import numpy as np
from str_generator import RegexStringGenerator
from IPython.display import clear_output
import time

gym.register(
    id="ComplexEnv-v0",
    entry_point="complex_problem_env:ComplexEnv",
)

class ComplexEnv(gym.Env):

    COMMUNICATE_COST = 1
    
    # symbol replacement
    #   a1 -> a  ,  e1 -> x
    #   a2 -> b  ,  e2 -> y
    global_transitions={
        0:{'a':1,  'b':2,  'x':0,  'y':0},
        1:{'b':3,  'x':2,  'y':1},
        2:{'a':4},
        3:{'a':4,  'b':1},
        4:{'x':5, 'y':0}
    }
    
    local_transitions={
        0:{'a':1,  'b':2,  'x':0,  'y':0},
        1:{'a':-1, 'b':3,  'x':2,  'y':1},
        2:{'a':4,  'b':-1, 'x':-1, 'y':-1},
        3:{'a':4,  'b':1,  'x':-1, 'y':-1},
        4:{'a':-1, 'b':-1, 'x':5, 'y':0},
        5:{'a':-1, 'b':-1, 'x':-1, 'y':-1},
        -1:{'a':-1, 'b':-1, 'x':-1, 'y':-1},
    }
    
    metadata = {'render.modes': ['human', 'simulation'], 'string.modes': ['full', 'half', 'simulation']}
    
    def __init__(self, string_mode="full", render_mode=None, max_star=5):
        self.string_generator = RegexStringGenerator(max_star=max_star)
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: don't communicate, 1:communicate
        self.observation_space = gym.spaces.Box(low=-1, high=5, shape=(3,), dtype=np.int32)
        
        assert string_mode is None or string_mode in self.metadata['string.modes']
        self.string_mode = string_mode
        
        assert render_mode is None or render_mode in self.metadata['render.modes']
        self.render_mode = render_mode
        
    def reset(self, seed=None, options=None):
        if self.string_mode == "full":
            self.string=self.string_generator.generate_full_training_str()+"$"
        elif self.string_mode == "half":
            self.string=self.string_generator.generate_half_training_str()+"$"
        elif self.string_mode == "simulation":
            self.string=self.string_generator.generate_simulation_str()+"$"
        self.string_index=0
        self.reward=0
        
        self.global_state = 0
        self.agent_1_observation = 0
        self.agent_2_observation = 0
        
        if self.render_mode == 'human':
            print(f"\nNew Episode: {self.string}")
            self.render()
            
        if self.render_mode == "simulation":
            print(f"\nSimulation String: {self.string}")
        
        curr_symbol=self.string[self.string_index]
        
        while curr_symbol in ['x','y']:
            if self.render_mode == 'human':
                self.render()
            if self.render_mode == "simulation":
                self.simulate(False)
            self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
            self.agent_1_observation = self.local_transitions[self.agent_1_observation].get(curr_symbol)
            self.agent_2_observation = self.local_transitions[self.agent_2_observation].get(curr_symbol)
            self.string_index += 1
            curr_symbol=self.string[self.string_index]

        
        config=(self.global_state, self.agent_1_observation, self.agent_2_observation)
        
        info={"input_alphabet":self.string[self.string_index]}
        return np.array(config, dtype=np.int32), info
        
    def step(self, action):

        agent_id, communicate = action
        info={}
        terminated = False
        truncated = False
        
        curr_symbol=self.string[self.string_index]
        
        self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
        if agent_id==1:
            self.agent_1_observation = self.local_transitions[self.agent_1_observation].get(curr_symbol)
            if communicate == 1:
                self.reward-=self.COMMUNICATE_COST
                self.agent_2_observation = self.local_transitions[self.agent_2_observation].get(curr_symbol)
        elif agent_id==2:
            self.agent_2_observation = self.local_transitions[self.agent_2_observation].get(curr_symbol)
            if communicate == 1:
                self.reward-=self.COMMUNICATE_COST
                self.agent_1_observation = self.local_transitions[self.agent_1_observation].get(curr_symbol)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
                
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate else 'did not communicate'} on '{curr_symbol}'")
            self.render()
        
        if self.render_mode == "simulation":
            self.simulate(False)
        
        self.string_index += 1
        
        curr_symbol=self.string[self.string_index]
        
        while curr_symbol in ['x','y']:
            if curr_symbol == 'x' and self.global_state == "4" and  (self.agent_1_observation == 4 or self.agent_2_observation == 4) and self.render_mode == "simulation":
                print("here")
                self.simulate(True)
            else:
                
                self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
                self.agent_1_observation = self.local_transitions[self.agent_1_observation].get(curr_symbol)
                self.agent_2_observation = self.local_transitions[self.agent_2_observation].get(curr_symbol)
                if self.render_mode == 'human':
                    self.render()
                elif self.render_mode == "simulation":
                    self.simulate(False)
            self.string_index += 1
            curr_symbol=self.string[self.string_index]
        
        # Penalty Assignment
        if self.global_state != 5 and ( self.agent_1_observation in [-1,5] and self.agent_2_observation in [-1,5]):
            self.reward -= 10
            terminated = True
        elif self.global_state == 5 and not(self.agent_1_observation == 5 or self.agent_2_observation == 5):
            self.reward -= 10
            terminated = True
        
        elif self.string[self.string_index]== "$" and self.global_state == 5 and (self.agent_1_observation == 5 or self.agent_2_observation == 5):
            terminated = True
            self.reward += 20
        
        elif self.string[self.string_index]=="$" and self.global_state == 4:
            truncated = True
        
        config = (self.global_state, self.agent_1_observation, self.agent_2_observation)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        return np.array(config, dtype=np.int32), self.reward, terminated, truncated, info
    
    def render(self):
        print(f"Current symbol: '{self.string[self.string_index]}'")
        print(f"Config: <{self.global_state}, {self.agent_1_observation}, {self.agent_2_observation}>")
    
    def simulate(self, disabled):
        a = [" ", " ", " ", " ", " "," "]

        a[self.global_state] = "*"

        block = "X" if disabled else "-"

        print(f"Config: <{self.global_state}, {self.agent_1_observation}, {self.agent_2_observation}>, Current symbol: '{self.string[self.string_index]}'")
        if disabled:
            if self.agent_1_observation == 4:
                print("Agent 1 disabled x")
            elif self.agent_2_observation == 4:
                print("Agent 2 disabled x")
                
        
        print(
         "            x,y                     ",
         "           /  |                     ",
         "           \  V                     ",
         "           + 0 +                    ",
        f"start ---->| {a[0]} | <----              ",
         "           + - +       \            ", 
         "           /   \        \           ",
         "        a /     \ b      \          ",
         "         V       V        \         ",
         "     + 1 +   x   + 2 +     \        ",
        f" y C | {a[1]} |------>| {a[2]} |      | y     ",
         "     + - +       + - +     /        ",
         "    /    ^         |      /         ",
         "  b \     \ b      | a   /          ",
         "     V    /        V    /           ",
         "     + 3 +       + 4 + /        + 5 +",
        f"     | {a[3]} |------>| {a[4]} | {block} {block} {block} >  | {a[5]} |",
         "     + - +   a   + - +    x     + - +",
        sep="\n")
        
        time.sleep(1)
        clear_output()
        print()
        
        
        
        