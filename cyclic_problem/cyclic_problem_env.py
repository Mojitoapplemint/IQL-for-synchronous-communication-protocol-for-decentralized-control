import gymnasium as gym
import numpy as np
from str_generator import RegexStringGenerator
from IPython.display import clear_output
import time

gym.register(
    id="CylicEnv-v0",
    entry_point="cyclic_problem_env:CylicEnv",
)

class CylicEnv(gym.Env):

    COMMUNICATE_COST = 10
    
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
    
    bottom_transitions={
        0:{'a':1,  'b':2,  'x':0,  'y':0},
        1:{'a':-1, 'b':3,  'x':2,  'y':1},
        2:{'a':4,  'b':-1, 'x':-1, 'y':-1},
        3:{'a':4,  'b':1,  'x':-1, 'y':-1},
        4:{'a':-1, 'b':-1, 'x':5, 'y':0},
        5:{'a':-1, 'b':-1, 'x':-1, 'y':-1},
        -1:{'a':-1, 'b':-1, 'x':-1, 'y':-1},
    }
    
    metadata = {'render_modes': ['human'], 'string_modes': ['training', 'simulation']}
    
    def __init__(self, string_mode="full", render_mode=None, max_star=5):
        self.string_generator = RegexStringGenerator(max_star=max_star)
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: communicate, 1:don't communicate
        self.observation_space = gym.spaces.Box(low=-1, high=5, shape=(3,), dtype=np.int32)
    
        assert string_mode in self.metadata['string_modes']
        self.string_mode = string_mode
        
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        
    def reset(self, seed=None, options=None):
        if self.string_mode == "training":
            self.string=self.string_generator.generate_training_str()+"$"
        elif self.string_mode == "simulation":
            self.string=self.string_generator.generate_simulation_str()+"$"
        self.string_index=0
        self.communication_count=0
        self.global_state = 0
        self.agent_1_belief = 0
        self.agent_2_belief = 0
        
        if self.render_mode == 'human':
            print(f"\nNew Episode: string_mode:{self.string_mode}, string: {self.string}")
            if self.string_mode == "simulation":
                print(f"\nStarting Simulation Mode. This is not a training episode.")
                self.simulate()
            else:
                self.render()
        
        curr_symbol=self.string[self.string_index]
        
        while curr_symbol in ['x','y']:
            self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
            self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
            self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
            self.string_index += 1
            curr_symbol=self.string[self.string_index]
            if self.render_mode == 'human':
                if self.string_mode == "simulation":
                    self.simulate()
                else:
                    self.render()

        
        config=(self.global_state, self.agent_1_belief, self.agent_2_belief)
        
        info={"input_alphabet":self.string[self.string_index], "string":self.string}
        return np.array(config, dtype=np.int32), info
        
    def step(self, action):
        if self.string_mode == "simulation":
            return self.simultaion_step(action)
        
        reward = 0
        agent_id, communicate = action
        info={}
        terminated = False
        truncated = False
        
        curr_symbol=self.string[self.string_index]
        
        self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
        if agent_id==1:
            self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
            if communicate == 1:
                reward-=self.COMMUNICATE_COST
                self.communication_count+=1
                self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
        elif agent_id==2:
            self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
            if communicate == 1:
                self.communication_count+=1
                reward-=self.COMMUNICATE_COST
                self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
                
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate==1 else 'did not communicate'} on '{curr_symbol}'")
            self.render()
        
        self.string_index += 1
        
        curr_symbol=self.string[self.string_index]
        
        while curr_symbol in ['x','y']:
            self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
            self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
            self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
            if self.render_mode == 'human':
                self.render()
            self.string_index += 1
            curr_symbol=self.string[self.string_index]
        
        # Penalty Assignment
        if self.global_state == 5 and not(self.agent_1_belief == 5 or self.agent_2_belief == 5):
            # Penalized configuration Condition 1
            reward -= 100
            terminated = True
        elif self.global_state != 5 and self.agent_1_belief == 5 and self.agent_2_belief == 5:
            # Penalized configuration Condition 2
            reward -= 100
            terminated = True
        
        if self.agent_1_belief == -1 and self.agent_2_belief == -1:
            # terminate current episode as soon as both agents are in dead state; shortening training time
            terminated = True
            reward -= 100
        
        elif self.string[self.string_index]== "$" and self.global_state == 5 and (self.agent_1_belief == 5 or self.agent_2_belief == 5):
            terminated = True
            reward += 200
        
        config = (self.global_state, self.agent_1_belief, self.agent_2_belief)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        return np.array(config, dtype=np.int32), reward, terminated, truncated, info
    
    def render(self):
        print(f"Current symbol: '{self.string[self.string_index]}'")
        print(f"Config: <{self.global_state}, {self.agent_1_belief}, {self.agent_2_belief}>")
    
    def simultaion_step(self, action):
        agent_id, communicate = action
        terminated = False
        simulation_result = False # whether the simulation ends in success or failure
        
        curr_symbol=self.string[self.string_index]
        self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
        if agent_id==1:
            self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
            if communicate == 1:
                self.communication_count+=1
                self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
        elif agent_id==2:
            self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
            if communicate == 1:
                self.communication_count+=1
                self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
        
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate==1 else 'did not communicate'} on '{curr_symbol}'")
            self.simulate()
            
        self.string_index += 1
        curr_symbol=self.string[self.string_index]
        
        
        if self.agent_1_belief == -1 and self.agent_2_belief == -1:
            terminated=True
            config = (self.global_state, self.agent_1_belief, self.agent_2_belief)
        
            info={"input_alphabet":self.string[self.string_index]}
        
            return np.array(config, dtype=np.int32), 0, terminated, simulation_result, info
        
        while curr_symbol in ['x','y']:
            if curr_symbol == 'y':
                self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
                self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
                self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
                self.string_index += 1
                curr_symbol=self.string[self.string_index]
                if self.render_mode == 'human':
                    self.simulate()
            else:
                if self.global_state == 4:
                    agent_1_disable_x = (self.agent_1_belief == 4)
                    agent_2_disable_x = (self.agent_2_belief == 4)     
                    
                    if agent_1_disable_x or agent_2_disable_x:
                        self.string_index += 1
                        curr_symbol=self.string[self.string_index]
                        if self.render_mode == 'human':
                            self.simulate(agent_1_disable_x, agent_2_disable_x)
                    
                    else:
                        self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
                        self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
                        self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
                        if self.render_mode == 'human':
                            print("Both agents failed to disable 'x' at state 4.")        
                            self.simulate()
                        terminated=True
                        break
                else:
                    self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
                    self.agent_1_belief = self.bottom_transitions[self.agent_1_belief].get(curr_symbol)
                    self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
                    self.string_index += 1
                    curr_symbol=self.string[self.string_index]
                    if self.render_mode == 'human':
                        self.simulate()

        
        if self.string[self.string_index]=="$" and self.global_state == 4:
            # condition for simulation where agent successfully disable x at state 4
            terminated = True
            simulation_result = True
        elif self.string[self.string_index]=="$":
            terminated = True
        
        config = (self.global_state, self.agent_1_belief, self.agent_2_belief)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        return np.array(config, dtype=np.int32), 0, terminated, simulation_result, info
        
        
    
    def simulate(self, agent_1_disable_x=False, agent_2_disable_x=False):
        a = [" ", " ", " ", " ", " "," "]

        a[self.global_state] = "#"

        block = "X" if (agent_1_disable_x or agent_2_disable_x) else "-"

        print(f"Config: <{self.global_state}, {self.agent_1_belief}, {self.agent_2_belief}>, Current symbol: '{self.string[self.string_index]}', # comm: {self.communication_count}")

        if agent_1_disable_x:
            print("Agent 1 is disabled 'c' at state:" , self.agent_1_belief)
        if agent_2_disable_x:
            print("Agent 2 is disabled 'c' at state:" , self.agent_2_belief)
        
                
        
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
        
        
        
        