import gymnasium as gym
import numpy as np
from str_generator import RegexStringGenerator

gym.register(
    id="ComplexEnv-v0",
    entry_point="complex_env:ComplexEnv",
)

class ComplexEnv(gym.Env):

    COMMUNICATE_COST = 1
    
    # symbol replacement
    # a1 -> a
    # a2 -> b
    # e1 -> x
    # e2 -> y
    
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
    
    metadata = {'render.modes': ['human', 'simulation']}
    
    
    def __init__(self, render_mode=None, max_star=10):
        self.string_generator = RegexStringGenerator(max_star=max_star)
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: don't communicate, 1:communicate
        self.observation_space = gym.spaces.Box(low=-1, high=5, shape=(3,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.metadata['render.modes']
        self.render_mode = render_mode
        
    def reset(self, seed=None, options=None):
        self.string=self.string_generator.generate_training_str()+"$"
        self.string_index=0
        self.reward=0
        
        self.global_state = 0
        self.agent_1_observation = 0
        self.agent_2_observation = 0
        
        config=(self.global_state, self.agent_1_observation, self.agent_2_observation)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        
        return np.array(config, dtype=np.int32), info
        
    def step(self, action):
        agent_id, communicate = action
        info={}
        terminated = False
        
        curr_symbol=self.string[self.string_index]
        
        self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
        if agent_id==1:
            self.agent_1_observation = self.agent_1_transitions[self.agent_1_observation].get(curr_symbol)
            if communicate == 1:
                self.reward-=self.COMMUNICATE_COST
                self.agent_2_observation = self.agent_2_transitions[self.agent_2_observation].get(curr_symbol)
        elif agent_id==2:
            self.agent_2_observation = self.agent_2_transitions[self.agent_2_observation].get(curr_symbol)
            if communicate == 1:
                self.reward-=self.COMMUNICATE_COST
                self.agent_1_observation = self.agent_1_transitions[self.agent_1_observation].get(curr_symbol)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
                
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate else 'did not communicate'} on '{curr_symbol}'")
            self.render()
        
        self.string_index += 1
        
        curr_symbol=self.string[self.string_index]
        
        while curr_symbol in ['x','y']:
            self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
            self.agent_1_observation = self.agent_1_transitions[self.agent_1_observation].get(curr_symbol)
            self.agent_2_observation = self.agent_2_transitions[self.agent_2_observation].get(curr_symbol)
            if self.render_mode == 'human':
                self.render()
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
        
        config = (self.global_state, self.agent_1_observation, self.agent_2_observation)
        return np.array(config, dtype=np.int32), self.reward, terminated, False, info
    
    def render(self):
        print(f"config: <{self.global_state}, {self.agent_1_observation}, {self.agent_2_observation}>")
        print(f"Current symbol: '{self.string[self.string_index]}'")