import gymnasium as gym
import numpy as np

gym.register(
    id="BenchmarkEnv-v0",
    entry_point="benchmark_env:BenchmarkEnv",
)

class BenchmarkEnv(gym.Env):
    COMMUNICATION_COST = 10
    
    global_transitions={
        1:{'a': 2, 'b': 3},
        2:{'b': 4},
        3:{'a': 5},
        4:{'s': 6},
        5:{'s': 7},
    }
    
    bottom_transitions={
        1:{'a': 2, 'b': 3, 's':-1},
        2:{'b': 4, 'a':-1, 's':-1},
        3:{'a': 5, 'b': -1, 's':-1},
        4:{'s': 6, 'a':-1, 'b':-1},
        5:{'s': 7, 'a':-1, 'b':-1},
    }
    
    metadata = {'render.modes': ['human']}
    
    L=['abs$', 'bas$']
    
    def __init__(self, render_mode=None):
        super(BenchmarkEnv, self).__init__()
        
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: don't communicate, 1:communicate
        self.observation_space = gym.spaces.Box(low=-1, high=7, shape=(3,), dtype=np.int32)  # States: 1 to 7, plus terminal state 0
        
        assert render_mode is None or render_mode in self.metadata['render.modes']
        self.render_mode = render_mode
        
        
    def reset(self, seed=None, options=None):
        self.string=self.L[np.random.randint(0, len(self.L))]
        #self.string=self.L[0]
        self.string_index=0
        self.reward=0
        
        self.global_state = 1
        self.agent_1_belief = 1
        self.agent_2_belief = 1
        
        config=(self.global_state, self.agent_1_belief, self.agent_2_belief)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        if self.render_mode == 'human':
            print(f"\n-------------New Episode: {self.string}---------------")
            self.render()
        return config, info
    
    def step(self, action):
        agent_id, communicate = action
        

        info={}
        
        curr_symbol=self.string[self.string_index]
        
        self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
        if agent_id==1:
            self.agent_1_belief = self.agent_1_transitions[self.agent_1_belief].get(curr_symbol)
            if communicate == 1:
                self.reward-=self.COMMUNICATION_COST
                self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
        elif agent_id==2:
            self.agent_2_belief = self.bottom_transitions[self.agent_2_belief].get(curr_symbol)
            if communicate == 1:
                self.reward-=self.COMMUNICATION_COST
                self.agent_1_belief = self.agent_1_transitions[self.agent_1_belief].get(curr_symbol)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
        
        self.string_index += 1
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate==1 else 'did not communicate'} on '{curr_symbol}'")
            self.render()
        
        if self.string[self.string_index]=='s':
            self.global_state=self.global_transitions[self.global_state].get('s')
            self.agent_1_belief=self.agent_1_transitions[self.agent_1_belief].get('s')
            self.agent_2_belief=self.bottom_transitions[self.agent_2_belief].get('s')

            config=(self.global_state, self.agent_1_belief, self.agent_2_belief)
            
            self.string_index+=1
            if self.render_mode == 'human':
                self.render()

            
            if config==(7,-1,-1) or config==(6,-1,-1):
                self.reward-=100
            else:
                self.reward+=100
        
        else:
            config=(self.global_state, self.agent_1_belief, self.agent_2_belief)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        return config, self.reward, None, None, info
            
    def render(self):
        print(f"Current Configuration<{self.global_state, self.agent_1_belief, self.agent_2_belief}>")
        print(f"Current Alphabet: {self.string[self.string_index]}")