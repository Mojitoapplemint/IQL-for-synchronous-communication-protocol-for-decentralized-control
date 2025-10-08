import gymnasium as gym
import numpy as np

gym.register(
    id="ComposedEnv-v0",
    entry_point="env:ComposedEnv",
)

class ComposedEnv(gym.Env):
    global_transitions={
        1:{'a': 2, 'b': 3},
        2:{'b': 4},
        3:{'a': 5},
        4:{'s': 6},
        5:{'s': 7},
    }
    
    agent_1_transitions={
        1:{'a': 2, 'b': 3},
        2:{'b': 4, 's':-1},
        3:{'a': 5},
        4:{'s': 6},
        5:{'s': 7},
    }
    
    agent_2_transitions={
        1:{'a': 2, 'b': 3},
        2:{'b': 4},
        3:{'a': 5, 's':-1},
        4:{'s': 6},
        5:{'s': 7},
    }
    
    metadata = {'render.modes': ['human']}
    
    L=['abs$', 'bas$']
    
    def __init__(self, render_mode=None):
        super(ComposedEnv, self).__init__()
        
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: don't communicate, 1:communicate
        self.observation_space = gym.spaces.Box(low=-1, high=7, shape=(3,), dtype=np.int32)  # States: 1 to 7, plus terminal state 0
        
        assert render_mode is None or render_mode in self.metadata['render.modes']
        self.render_mode = render_mode
        
        
    def reset(self, seed=None, options=None):
        # self.string=self.L[np.random.randint(0, len(self.L))]
        self.string=self.L[0]
        self.string_index=0
        
        self.global_state = 1
        self.agent_1_observation = 1
        self.agent_2_observation = 1
        
        config=(self.global_state, self.agent_1_observation, self.agent_2_observation)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        if self.render_mode == 'human':
            print(f"\n-------------New Episode: {self.string}---------------")
            self.render()
        return config, info
    
    def step(self, action):
        agent_id, communicate = action
        
        reward=0
        terminated=False
        truncated=False
        info={}
        
        alphabet=self.string[self.string_index]
        
        self.global_state = self.global_transitions[self.global_state].get(alphabet)
        if agent_id==1:
            self.agent_1_observation = self.agent_1_transitions[self.agent_1_observation].get(alphabet)
            if communicate == 1:
                reward-=1
                self.agent_2_observation = self.agent_2_transitions[self.agent_2_observation].get(alphabet)
        elif agent_id==2:
            self.agent_2_observation = self.agent_2_transitions[self.agent_2_observation].get(alphabet)
            if communicate == 1:
                reward-=1
                self.agent_1_observation = self.agent_1_transitions[self.agent_1_observation].get(alphabet)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
        
        self.string_index += 1
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate else 'did not communicate'} on '{alphabet}'")
            self.render()
        
        if self.string[self.string_index]=='s':
            self.global_state=self.global_transitions[self.global_state].get('s')
            self.agent_1_observation=self.agent_1_transitions[self.agent_1_observation].get('s')
            self.agent_2_observation=self.agent_2_transitions[self.agent_2_observation].get('s')

            config=(self.global_state, self.agent_1_observation, self.agent_2_observation)
            
            self.string_index+=1
            if self.render_mode == 'human':
                self.render()

            
            if config==(7,-1,-1) or config==(6,-1,-1):
                reward-=10
            else:
                reward+=10
            terminated=True
        
        else:
            config=(self.global_state, self.agent_1_observation, self.agent_2_observation)
        
        info={"input_alphabet":self.string[self.string_index]}
        
        return config, reward, terminated, truncated, info
            
    def render(self):
        print(f"Current Configuration<{self.global_state, self.agent_1_observation, self.agent_2_observation}>")
        print(f"Current Alphabet: {self.string[self.string_index]}")