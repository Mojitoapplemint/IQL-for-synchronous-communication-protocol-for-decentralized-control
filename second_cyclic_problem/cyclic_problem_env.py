import gymnasium as gym
import numpy as np
from word_generator import RegexWordGenerator
from IPython.display import clear_output
import time
import pandas as pd

gym.register(
    id="CyclicEnv2-v0",
    entry_point="cyclic_problem_env:CylicEnv",
)

class CylicEnv(gym.Env):

    COMMUNICATE_COST = 10
    
    global_transitions={
        1:{'a': 2, 'b': 5},
        2:{'b': 3},
        3:{'c':1, 's': 4},
        5:{'a': 6},
        6:{'c':1,'s':7},
    }
    
    m_bottom={
        1:{1},
        2:{2},
        3:{1,3},
        4:{4},
        5:{5},
        6:{6,1},
        7:{7},
        -1:{-1},
    }
    
    agent_0_transitions={
        1:{'a': 2, 'b': 5},
        2:{'b': 3},
        3:{'a':2, 'b': 5, 's': 4},
        5:{'a': 6},
        6:{'a':2, 'b':5, 's':7},
    }
    
    bottom_transitions={
        1:{'a': 2, 'b': 5, 's':-1},
        2:{'b': 3, 'a':-1, 's':-1},
        3:{'a': 2, 'b': 5, 's': 4},
        5:{'a': 6, 'b':-1, 's':-1},
        6:{'a': 2, 'b': 5, 's': 7},
        4:{'s':-1, 'a':-1, 'b':-1},
        7:{'s':-1, 'a':-1, 'b':-1},
       -1:{'s':-1, 'a':-1, 'b':-1},
    }
    
    metadata = {'render_modes': ['human'], 'string_modes': ['training', 'simulation', 'stats']}
    
    def __init__(self, string_mode="full", render_mode=None, max_star=3):
        self.string_generator = RegexWordGenerator(max_star=max_star)
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: communicate, 1:don't communicate
        self.observation_space = gym.spaces.Box(low=-1, high=5, shape=(3,), dtype=np.int32)
    
        assert string_mode in self.metadata['string_modes']
        self.string_mode = string_mode
        
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        
        if self.string_mode == 'stats':
            df = pd.read_csv("second_cyclic_problem/strings.csv")
            self.string_list = df["strings"].to_list()
            self.index = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.string_mode == "training":
            self.string=self.string_generator.generate_training_word()+"$"
        elif self.string_mode == "simulation":
            self.string=self.string_generator.generate_simulation_word()+"$"
        elif self.string_mode == "stats":
            self.string=self.string_list[self.index]+"$"
            self.index = (self.index + 1) % len(self.string_list)
        self.string_index=0
        self.communication_count=0
        self.agent_0_state = 1
        self.agent_1_belief = 1
        self.agent_2_belief = 1
        
        if self.render_mode == 'human':
            print(f"\nNew Episode: string_mode:{self.string_mode}, string: {self.string}")
            if self.string_mode == "simulation":
                print(f"\nStarting Simulation Mode. This is not a training episode.")
                self.simulate()
            else:
                self.render()
        
        curr_symbol=self.string[self.string_index]
        
        config=(self.agent_0_state, self.agent_1_belief, self.agent_2_belief)
        
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
        
        self.agent_0_state = self.agent_0_transitions[self.agent_0_state].get(curr_symbol)
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
        
        if curr_symbol=='s':
            self.agent_0_state=self.agent_0_transitions[self.agent_0_state].get('s')
            self.agent_1_belief=self.bottom_transitions[self.agent_1_belief].get('s')
            self.agent_2_belief=self.bottom_transitions[self.agent_2_belief].get('s')
        
            if self.render_mode == 'human':
                self.render()
            self.string_index += 1
            curr_symbol=self.string[self.string_index]
            
        # Penalty Assignment
        if self.agent_1_belief == -1 and self.agent_2_belief == -1:
            # terminate current episode as soon as both agents are in dead state; shortening training time
            terminated = True
            reward -= 200
        if self.agent_0_state ==4 and not(self.agent_1_belief == 4 or self.agent_2_belief == 4):
            # Penalized configuration Condition 1
            reward -= 200
            terminated = True
        elif self.agent_0_state ==7 and not(self.agent_1_belief == 7 or self.agent_2_belief == 7):
            # Penalized configuration Condition 1
            reward -= 200
            terminated = True
        elif self.agent_0_state != 4 and self.agent_1_belief in [4,-1] and self.agent_2_belief in [4, -1]:
            # Penalized configuration Condition 2
            reward -=  200
            terminated = True
        elif self.agent_0_state != 7 and self.agent_1_belief in [7,-1] and self.agent_2_belief in [7, -1]:
            # Penalized configuration Condition 2
            reward -=  200
            terminated = True
        elif self.string[self.string_index]=="$":
            terminated = True
        
        config = (self.agent_0_state, self.agent_1_belief, self.agent_2_belief)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        return np.array(config, dtype=np.int32), reward, terminated, truncated, info
    
    def render(self):
        print(f"Current symbol: '{self.string[self.string_index]}'")
        print(self.agent_0_state,self.agent_1_belief, self.agent_2_belief)
        print(f"Config: <{self.agent_0_state}:{self.m_bottom[self.agent_0_state]}, {self.agent_1_belief}:{self.m_bottom[self.agent_1_belief]}, {self.agent_2_belief}:{self.m_bottom[self.agent_2_belief]}>")
         
    def simultaion_step(self, action):
        # Note: In simulation mode, we still use variable "agent_0_state", but this refers to the actual global state.
        agent_id, communicate = action
        terminated = False
        simulation_result = False # whether the simulation ends in success or failure
        
        curr_symbol=self.string[self.string_index]
        self.agent_0_state = self.global_transitions[self.agent_0_state].get(curr_symbol)
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
            config = (self.agent_0_state, self.agent_1_belief, self.agent_2_belief)
        
            info={"input_alphabet":self.string[self.string_index]}
        
            return np.array(config, dtype=np.int32), 0, terminated, simulation_result, info
        
        if curr_symbol=='s':
            if self.agent_0_state ==6:
                self.agent_0_state=self.global_transitions[self.agent_0_state].get('s')
                self.agent_1_belief=self.bottom_transitions[self.agent_1_belief].get('s')
                self.agent_2_belief=self.bottom_transitions[self.agent_2_belief].get('s')
                if self.render_mode == 'human':
                    self.simulate()
                self.string_index += 1
                curr_symbol=self.string[self.string_index]
                
            elif self.agent_0_state ==3:
                agent_1_disable_s = (self.agent_1_belief ==3)
                agent_2_disable_s = (self.agent_2_belief ==3)
                
                if agent_1_disable_s or agent_2_disable_s:
                    if self.render_mode == 'human':
                        self.simulate(agent_1_disable_s, agent_2_disable_s)
                    self.string_index+=1
                    curr_symbol=self.string[self.string_index]               
                
                else:
                    self.agent_0_state=self.global_transitions[self.agent_0_state].get('s')
                    self.agent_1_belief=self.bottom_transitions[self.agent_1_belief].get('s')
                    self.agent_2_belief=self.bottom_transitions[self.agent_2_belief].get('s')

                    if self.render_mode == 'human':
                        print("Both agents failed to disable 's' in state 3")
                        self.simulate(agent_1_disable_s, agent_2_disable_s)
                    terminated = True
                    
        if curr_symbol == 'c':
            self.agent_0_state = self.global_transitions[self.agent_0_state].get('c')
            if self.render_mode == 'human':
                self.simulate()
            self.string_index += 1
            curr_symbol=self.string[self.string_index]
        
        
        if curr_symbol == '$' and (self.agent_0_state == self.agent_1_belief or self.agent_0_state == self.agent_2_belief):
            simulation_result = True
            terminated = True

        if curr_symbol=="$":
            terminated = True
        
        
        config = (self.agent_0_state, self.agent_1_belief, self.agent_2_belief)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        return np.array(config, dtype=np.int32), 0, terminated, simulation_result, info
        
        
    
    def simulate(self, agent_1_disable_s=False, agent_2_disable_s=False):
        a = [" " for _ in range(7)]
        a[self.agent_0_state-1] = "#"
        
        print(f"global state: {self.agent_0_state},\n agent 1's belief: {self.agent_1_belief}:{self.m_bottom[self.agent_1_belief]},\n agent 2's belief: {self.agent_2_belief}:{self.m_bottom[self.agent_2_belief]}>,\n Current symbol: '{self.string[self.string_index]}', # comm: {self.communication_count}")
        
        block = "|"
        
        if agent_1_disable_s or agent_2_disable_s:
            block = "X"
        
        
        print(
                 "\n            start             \n"
                 "              |               \n"
                 "              V               \n"
                 "            +-1-+ \n"
                f"    ------->| {a[0]} |<------- \n"
                 "    |       +---+       | \n"
                 "    |        / \        | \n"
                 "    |     a /   \ b     | \n"
                 "    |      V     V      | \n"
                 "    |   +-5-+   +-2-+   | \n"
                f"  c |   | {a[4]} |   | {a[1]} |   | c \n"
                 "    |   +---+   +---+   | \n"
                 "    |   a |       | b   | \n"
                 "    |     V       V     | \n"
                 "    |   +-6-+   +-3-+   | \n"
                f"    ----| {a[5]} |   | {a[2]} |---- \n"
                 "        +---+   +---+ \n"
                f"          |       {block} \n"
                f"        s |       {block} s \n"
                 "          V       V \n"
                 "        +-7-+   +-4-+ \n"
                f"        | {a[6]} |   | {a[3]} | \n"
                 "        +---+   +---+ \n"
                )
        time.sleep(1)
        print("" , end="\r")
