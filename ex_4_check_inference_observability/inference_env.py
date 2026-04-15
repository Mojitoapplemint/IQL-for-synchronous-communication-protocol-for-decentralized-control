import gymnasium as gym
import numpy as np
import time
from word_generator import WordGenerator
from IPython.display import clear_output
import pandas as pd

gym.register(
    id='InferenceEnv-v1',
    entry_point='inference_env:InferenceEnv',
)

class InferenceEnv(gym.Env):
    COMMUNICATION_COST = 30
    D_PENALTY = 100
    E_PENALTY = 0
    
    D_PEN_STATES = {17,18,21}
    
    E_PEN_STATES = {16,19,20}
    
    STATES_DISABLE_SIGMA = {3,8,13}
    
    m_L_transitions = {
        0: {"s": 0, "a": 1, "b": 5, "c": 10},
        1: {"s": 1, "p": 15, "m": 2},
        2: {"s": 2, "p": 3, "m": 4},
        3: {"s": 17},
        4: {"s": 16},
        5: {"s": 5, "q": 6, "r": 7},
        6: {"s": 6, "m": 8},
        7: {"s": 7, "m": 9},
        8: {"s": 18},
        9: {"s": 19},
        10: {"s": 10, "p": 11, "q": 12},
        11: {"s": 11, "m": 13},
        12: {"s": 12, "m": 14},
        13: {"s": 21},
        14: {"s": 20},
        15: {"s": 15, "m": 4},
    }
    
    m_L_bot_transitions = {
        0:  {'': 0,  "s": 0, "a": 1, "b": 5, "c": 10,     "p": -1, "m": -1, "q": -1, "r": -1},
        1:  {'': 1,  "s": 1, "p": 15, "m": 2,                "a": -1, "b": -1, "c": -1, "q": -1, "r": -1},
        2:  {'': 2,  "s": 2, "p": 3, "m": 4,                 "a": -1, "b": -1, "c": -1, "q": -1, "r": -1},
        3:  {'': 3,  "s": 17,                                "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        4:  {'': 4,  "s": 16,                                "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        5:  {'': 5,  "s": 5, "q": 6, "r": 7,                 "a": -1, "b": -1, "c": -1, "p": -1, "m": -1},
        6:  {'': 6,  "s": 6, "m": 8,                         "a": -1, "b": -1, "c": -1, "p": -1, "q": -1, "r": -1},
        7:  {'': 7,  "s": 7, "m": 9,                         "a": -1, "b": -1, "c": -1, "p": -1, "q": -1, "r": -1},
        8:  {'': 8,  "s": 18,                                "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        9:  {'': 9,  "s": 19,                                "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        10: {'': 10, "s": 10, "p": 11, "q": 12,              "a": -1, "b": -1, "c": -1, "m": -1, "r": -1},
        11: {'': 11, "s": 11, "m": 13,                       "a": -1, "b": -1, "c": -1, "p": -1, "q": -1, "r": -1},
        12: {'': 12, "s": 12, "m": 14,                       "a": -1, "b": -1, "c": -1, "p": -1, "q": -1, "r": -1},
        13: {'': 13, "s": 21,                                "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        14: {'': 14, "s": 20,                                "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        15: {'': 15, "s": 15, "m": 4,                        "a": -1, "b": -1, "c": -1, "p": -1, "q": -1, "r": -1},
        16: {'': 16, "s": -1, "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        17: {'': 17, "s": -1, "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        18: {'': 18, "s": -1, "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        19: {'': 19, "s": -1, "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        20: {'': 20, "s": -1, "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        21: {'': 21, "s": -1, "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
        -1: {'': -1, "s": -1, "a": -1, "b": -1, "c": -1, "p": -1, "m": -1, "q": -1, "r": -1},
    }
    
    meta_data = {'render_modes':['human'], 'string_modes':['training', 'simulation']}
    
    def __init__(self, render_mode=None, string_mode='training', max_star=3):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=22, shape=(3,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.meta_data['render_modes'], f"Invalid render mode: {render_mode}"
        assert string_mode in self.meta_data['string_modes'], f"Invalid string mode: {string_mode}"
        
        self.render_mode = render_mode
        self.string_mode = string_mode
        
        self.word_generator = WordGenerator(max_star=max_star)
        
        self.simulation_words = pd.read_csv('check_inference_observability/simulation_words.csv')['word'].tolist()
        # self.simulation_words_index = np.random.randint(0, len(self.simulation_words)-1)
        self.simulation_words_index = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.system_state, self.agent_1_state, self.agent_2_state= 0,0,0
        
        self.word_index=0
        
        if self.string_mode == 'training':
            self.word = self.word_generator.generate_training_word()+'$'
            if self.render_mode == 'human':
                print(f"\n===========New Episode=========== \nTraining word: {self.word}")
                print(f"Initial V structure state: <{self.m_L_states[self.system_state]}, {self.m_L_states[self.agent_1_state]}, {self.m_L_states[self.agent_2_state]}>")
                
            
        elif self.string_mode == 'simulation':
            self.word = self.simulation_words[self.simulation_words_index] + '$'
            self.simulation_words_index = (self.simulation_words_index + 1)%len(self.simulation_words)
            self.curr_event = self.word[self.word_index]
                        
            if self.render_mode == 'human':
                print(f"\n===========New Simulation=========== \nSimulation word: {self.word}")
                self.simulate()
        
        self.curr_event = self.word[self.word_index]

        while self.curr_event == 's':
            self.v_transition(['s', 's', 's'])
            if self.render_mode == 'human':
                self.render()
            
            # Updating current event to '$' indicating end of the word
            self.word_index += 1
            self.curr_event = self.word[self.word_index]
                
        v_state = (self.system_state, self.agent_1_state, self.agent_2_state)
        info  = {'word': self.word, 'curr_event': self.curr_event}
        
        return v_state, info
                
    def v_transition(self, vector_label):
        
        self.system_state = self.m_L_transitions[self.system_state][vector_label[0]]
        self.agent_1_state = self.m_L_bot_transitions[self.agent_1_state][vector_label[1]]
        self.agent_2_state = self.m_L_bot_transitions[self.agent_2_state][vector_label[2]]


    def step(self, action):
        agent_id, communicate = action
        reward = 0
        terminated = False
        simulation_result = False
        
        vector_label = [self.curr_event,'','']

        communication_cost = 0
        
        if agent_id == 1:
            vector_label[1] = self.curr_event
            
            if communicate:
                vector_label[2] = self.curr_event
                communication_cost -= self.COMMUNICATION_COST
        else:
            vector_label[2] = self.curr_event
            
            if communicate:
                vector_label[1] = self.curr_event
                communication_cost -= self.COMMUNICATION_COST

        
        # Performing the V structure transition based on the vector label, which is determined by the current event and the communication actions of the agents
        self.v_transition(vector_label)
        
        # Rendering the environment if in human mode
        if self.render_mode == 'human':
            agents = [1,2]
            
            agents.remove(agent_id)
            
            receive_1 = vector_label[agents[0]]==self.curr_event
            
            self.render(agent_id, receive_1, agents[0])
            
            if self.string_mode == 'simulation':
                self.simulate()
        
        # Updating the current event to the next event in the word
        self.word_index += 1
        self.curr_event = self.word[self.word_index]
        
        
        # State transition for 's' for training mode
        while self.curr_event == 's' and self.string_mode == 'training':
            self.v_transition(['s', 's', 's'])
            if self.render_mode == 'human':
                self.render()
            
            # Updating current event to '$' indicating end of the word
            self.word_index += 1
            self.curr_event = self.word[self.word_index]
            
        # State transition for 's' for simulation
        while self.curr_event == 's' and (self.string_mode == 'simulation'):
            
            # If agent 2 think it is in state where s must be disabled, then it disables it
            agent_2_disable = self.agent_2_state in self.STATES_DISABLE_SIGMA
            agent_1_disable = self.agent_1_state in self.STATES_DISABLE_SIGMA
            
            is_disabled = agent_2_disable or agent_1_disable
            
            if not is_disabled:
                self.v_transition(['s', 's', 's'])
                
            if self.render_mode == 'human':
                self.render()
                self.simulate(agent_1_disable = agent_1_disable, agent_2_disable=agent_2_disable)
            
            # Check simulation results based on system state and agent 2's disable status
            if (self.system_state in self.E_PEN_STATES) and not is_disabled:
                simulation_result = True

            if (self.system_state in self.STATES_DISABLE_SIGMA) and is_disabled:
                simulation_result = True
                
            # Updating current event to '$' indicating end of the word
            self.word_index += 1
            self.curr_event = self.word[self.word_index]
        
        
        if (self.system_state in self.D_PEN_STATES) and self.agent_1_state == -1 and self.agent_2_state == -1:
            terminated = True
            reward = -self.D_PENALTY
        if (self.system_state in self.E_PEN_STATES) and self.agent_1_state == -1 and self.agent_2_state == -1:
            terminated = True
            reward = -self.E_PENALTY
        elif self.curr_event == '$':
            terminated = True
        

        v_state = (self.system_state, self.agent_1_state, self.agent_2_state)
        info = {'curr_event': self.curr_event}
        
        if self.string_mode == 'simulation' or self.string_mode == 'stats':
            truncated = simulation_result
        else:
            truncated = False
        
        return np.array(v_state, dtype=np.int32), (communication_cost, reward), terminated, truncated, info
        
    def render(self, sender_1=None, communicate_1=None, receiver_1=None):
        print(f"\nEvent {self.curr_event} occurred")
        
        if self.curr_event not in ['d', 's']:
            print(f"Agent {sender_1} {'communicated' if communicate_1 else 'did not communicate'} '{self.curr_event}' to Agent {receiver_1}")


        if self.string_mode == 'training':
            print(f"Resulting V structure state: <{self.system_state}, {self.agent_1_state}, {self.agent_2_state}>")
        elif self.string_mode == 'simulation':
            print(f"Resulting V structure state: <{self.system_state}, {self.agent_1_state}, {self.agent_2_state}>")

    
    def simulate(self, agent_1_disable=False, agent_2_disable=False):
        a = [" " for _ in range(22)]
        a[self.system_state] = "#"
        
        block = "-"

        
        if self.curr_event == 's':
            if agent_2_disable:
                print("Agent 2 disabled 's'")
                block = "X"
            
            if agent_1_disable:
                print("Agent 1 disabled 's'")
                block = "X"
                 
            if self.system_state in self.E_PEN_STATES and not (agent_2_disable or agent_1_disable):
                print("\nSuccessfully enabled s")
            
            if self.system_state in self.D_PEN_STATES and (agent_2_disable or agent_1_disable):
                print("\nFailed to enable s")
            
            if self.system_state == self.STATES_DISABLE_SIGMA and (agent_2_disable or agent_1_disable):
                print("\nSuccessfully disabled s")
                
            if self.system_state == self.D_PEN_STATES and not (agent_2_disable or agent_1_disable):
                print("\nFailed to disable s")
        
        print(
        "                                      s __                                    \n"
        "                                       |  v                                   \n"
        "                   s __          p    +-15-+   m    +-04-+   s    +-16-+       \n"
        f"                    |  v      /-----> |  {a[15]} | -----> |  {a[4]} | {block}-{block}-{block}> |  {a[16]} |       \n"
        "                   +-01-+    /        +----+     -> +----+        +----+       \n"
        f"             /---> |  {a[1]} | --+                 m /                              \n"
        "         a  /      +----+    \\        +-02-+ ---    +-03-+        +-17-+       \n"
        f"           /                  \\-----> |  {a[2]} | -----> |  {a[3]} | {block} {block} {block}> |  {a[17]} |       \n"
        "          /                      m    +----+   p    +----+   s    +----+       \n"
        "         |                             \\  ^                                    \n"
        "         |                              \\/ s                                  \n"
        "         |                                                                    \n"
        "         |                            s __                                    \n"
        "         |                             |  v                                   \n"
        "   s __  |         s __          q    +-06-+        +-08-+   s    +-18-+       \n"
        f"    |  v |          |  v      /-----> |  {a[6]} | -----> |  {a[8]} | {block} {block} {block}> |  {a[18]} |       \n"
        "     +-00-+   b    +-05-+    /        +----+        +----+        +----+       \n"
        f"     |  {a[0]} | -----> |  {a[5]} | --+                                                  \n"
        "     +----+        +----+    \\        +-07-+        +-09-+        +-19-+       \n"
        f"         |                    \\-----> |  {a[7]} | -----> |  {a[9]} | {block}-{block}-{block}> |  {a[19]} |       \n"
        "         |                       r    +----+        +----+   s    +----+       \n"
        "         |                             \\  ^                                     \n"
        "         |                              \\/ s                                     \n"
        "         |                                                                     \n"
        "         |                            s __                                     \n"
        "         |                             |  v                                    \n"
        "          \\        s __          q    +-12-+   m    +-14-+   s    +-20-+       \n"
        f"           \\        |  v      /-----> |  {a[12]} | -----> |  {a[14]} | {block}-{block}-{block}> |  {a[20]} |       \n"
        "         c  \\      +-10-+    /        +----+        +----+        +----+       \n"
        f"             \\---> |  {a[10]} | --+                                                  \n"
        "                   +----+    \\        +-11-+        +-13-+        +-21-+       \n"
        f"                              \\-----> |  {a[11]} | -----> |  {a[13]} | {block} {block} {block}> |  {a[21]} |       \n"
        "                                 p    +----+   m    +----+   s    +----+       \n"
        "                                       \\  ^                                    \n"
        "                                        \\/ s                                    \n"
        )

        time.sleep(1)
        clear_output()
        print()