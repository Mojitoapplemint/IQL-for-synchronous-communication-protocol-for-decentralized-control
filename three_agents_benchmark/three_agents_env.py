import gymnasium as gym
import numpy as np
import time
from IPython.display import clear_output

gym.register(
    id="ThreeAgentsEnv-v0",
    entry_point="three_agents_env:ThreeAgentsEnv",
)

class ThreeAgentsEnv(gym.Env):
    COMMUNICATE_COST = 50
    
    m_L_transitions = {
        1: {'a':2, 'b':7, 'c':12},
        2: {'b':3, 'c':4},
        3: {'s':5},
        4: {'s':6},
        7: {'a':9, 'c':8},
        8: {'s':10},
        9: {'s':11},
        12: {'a':13, 'b':14},
        13: {'s':15},
        14: {'s':16},
    }
    
    m_L_bot_transitions = {
        1:  {'':1 , 'a':2, 'b':7, 'c':12, 's':-1},
        2:  {'':2 , 'b':3, 'c':4,         'a':-1, 's':-1},
        3:  {'':3 , 's':5,                'a':-1, 'b':-1, 'c':-1},
        4:  {'':4 , 's':6,                'a':-1, 'b':-1, 'c':-1},
        5:  {'':5 ,                       's':-1,'a':-1, 'b':-1, 'c':-1},
        6:  {'':6 ,                       's':-1,'a':-1, 'b':-1, 'c':-1},
        7:  {'':7 , 'a':9, 'c':8,         'b':-1, 's':-1},
        8:  {'':8 , 's':10,               'a':-1, 'b':-1, 'c':-1},
        9:  {'':9 , 's':11,               'a':-1, 'b':-1, 'c':-1},
        10: {'':10,                       's':-1,'a':-1, 'b':-1, 'c':-1},
        11: {'':11,                       's':-1,'a':-1, 'b':-1, 'c':-1},
        12: {'':12, 'a':13, 'b':14,       'c':-1, 's':-1},
        13: {'':13, 's':15,               'a':-1, 'b':-1, 'c':-1},
        14: {'':14, 's':16,               'a':-1, 'b':-1, 'c':-1},
        15: {'':15,                       's':-1,'a':-1, 'b':-1, 'c':-1},
        16: {'':16,                       's':-1,'a':-1, 'b':-1, 'c':-1},
    }
    
    metadata = {'render_modes': ['human', 'simulation']}
    
    L_tilde = ["abs", "acs", "bas", "bcs", "cas", "cbs"]
    
    def __init__(self, render_mode=None):
        self.action_space = gym.spaces.Discrete(4) 
        self.observation_space = gym.spaces.Box(low=-1, high=16, shape=(4,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.metadata['render_modes']
        
        self.render_mode = render_mode
        self.training_word_selection = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.system_state = 1
        self.agent_1_state = 1
        self.agent_2_state = 1
        self.agent_3_state = 1
        self.training_word_index = 0
        
        self.input_word = self.L_tilde[self.training_word_selection]+"$"
        
        self.training_word_selection = (self.training_word_selection + 1) % len(self.L_tilde)
        
        if self.render_mode == 'human':
            print("\n======== New Epoch =========")
            print("Training String: ", self.input_word, "\n")
            self.render()
        elif self.render_mode == 'simulation':
            print("Simulation String: ", self.input_word)
            self.simulate()

        config = (self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state)

        curr_event=self.input_word[self.training_word_index]

        info = {'curr_event': curr_event, "string": self.input_word}
        
        return config, info
    
    def step(self, action):
        if self.render_mode == 'simulation':
            return self.simulation_step(action)
        reward = 0
        agent_id, communicate = action
        terminated = False
        
        curr_event = self.input_word[self.training_word_index]
        
        vector_label = [curr_event if communicate[i]==1 else '' for i in range(4)]
        
        self.system_state = self.m_L_transitions[self.system_state][vector_label[0]]
        
        self.agent_1_state = self.m_L_bot_transitions[self.agent_1_state][vector_label[1]]
        
        self.agent_2_state = self.m_L_bot_transitions[self.agent_2_state][vector_label[2]]
        
        self.agent_3_state = self.m_L_bot_transitions[self.agent_3_state][vector_label[3]]
        
        communication_cost = -1* sum(communicate) * self.COMMUNICATE_COST
        
        if self.render_mode == 'human' :
            temp = [1,2,3]
            events = ['a','b','c']
            
            temp.remove(agent_id)
                        
            receive_1 = communicate[temp[0]]
            receive_2 = communicate[temp[1]]
            
            print(f"Agent {agent_id} {'communicated' if receive_1 else 'did not communicate'} '{events[agent_id-1]}' to Agent {temp[0]}")
            print(f"Agent {agent_id} {'communicated' if receive_2 else 'did not communicate'} '{events[agent_id-1]}' to Agent {temp[1]}")
            
            print(f"\nEvent '{self.input_word[self.training_word_index]}' occured")
            self.render()
        
        self.training_word_index += 1
        
        curr_event=self.input_word[self.training_word_index]
        
        if curr_event == 's':
            self.system_state = self.m_L_transitions[self.system_state]['s']
        
            self.agent_1_state = self.m_L_bot_transitions[self.agent_1_state]['s']
        
            self.agent_2_state = self.m_L_bot_transitions[self.agent_2_state]['s']
        
            self.agent_3_state = self.m_L_bot_transitions[self.agent_3_state]['s']
        
            if self.render_mode == 'human':  
                print(f"\nEvent '{self.input_word[self.training_word_index]}' occured")              
                self.render()
            
            self.training_word_index += 1
            curr_event=self.input_word[self.training_word_index] 
        
        
        if self.system_state in [5,6,10,11,15,15] and self.agent_1_state == -1 and self.agent_2_state == -1 and self.agent_3_state == -1:
            terminated = True
            reward -= 100
        
        elif curr_event == '$':
            terminated = True
        
        
        config = (self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state)

        curr_event=self.input_word[self.training_word_index]

        info = {'curr_event': curr_event, "string": self.input_word}
        
        return np.array(config, dtype=np.int32), (reward, communication_cost), terminated, False, info

    def render(self):
        print(f"Resulting V state: ({self.system_state}, {self.agent_1_state}, {self.agent_2_state}, {self.agent_3_state})")
    
    def simulation_step(self, action):
        reward = 0
        agent_id, communicate = action
        terminated = False
        
        curr_event = self.input_word[self.training_word_index]
        
        vector_label = [curr_event if communicate[i]==1 else '' for i in range(4)]
        
        self.system_state = self.m_L_transitions[self.system_state][vector_label[0]]
        
        self.agent_1_state = self.m_L_bot_transitions[self.agent_1_state][vector_label[1]]
        
        self.agent_2_state = self.m_L_bot_transitions[self.agent_2_state][vector_label[2]]
        
        self.agent_3_state = self.m_L_bot_transitions[self.agent_3_state][vector_label[3]]
        
        communication_cost = sum(communicate) * self.COMMUNICATE_COST
        

        temp = [1,2,3]
        events = ['a','b','c']
        
        temp.remove(agent_id)
                    
        receive_1 = communicate[temp[0]]
        receive_2 = communicate[temp[1]]

        self.render()
        print(f"Agent {agent_id} {'communicated' if receive_1 else 'did not communicate'} '{events[agent_id-1]}' to Agent {temp[0]}")
        print(f"Agent {agent_id} {'communicated' if receive_2 else 'did not communicate'} '{events[agent_id-1]}' to Agent {temp[1]}")
        
        self.simulate()
        
        self.training_word_index += 1
        
        curr_event=self.input_word[self.training_word_index]
        
        
        if curr_event == 's':
            agent_1_disable = self.agent_1_state in [3,8,13]
            agent_2_disable = self.agent_2_state in [3,8,13]
            agent_3_disable = self.agent_3_state in [3,8,13]
            
            if not (agent_1_disable or agent_2_disable or agent_3_disable):
            
                self.system_state = self.m_L_transitions[self.system_state]['s']
            
                self.agent_1_state = self.m_L_bot_transitions[self.agent_1_state]['s']
            
                self.agent_2_state = self.m_L_bot_transitions[self.agent_2_state]['s']
            
                self.agent_3_state = self.m_L_bot_transitions[self.agent_3_state]['s']
            
            self.render()
            self.simulate(agent_1_disable=agent_1_disable, agent_2_disable=agent_2_disable, agent_3_disable=agent_3_disable)
            
            self.training_word_index += 1
            curr_event=self.input_word[self.training_word_index] 

        if curr_event == '$':
            terminated = True
        
        
        config = (self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state)

        curr_event=self.input_word[self.training_word_index]

        info = {'curr_event': curr_event, "string": self.input_word}
        
        return np.array(config, dtype=np.int32), (reward, communication_cost), terminated, False, info

    
    def simulate(self, agent_1_disable=False, agent_2_disable=False, agent_3_disable=False):
        
        a = [" " for _ in range(17)]
        a[self.system_state] = "#"        
        
        
        block_v = "|"
        block_h = "-"
        
        if self.input_word[self.training_word_index] == 's':
            
            if agent_1_disable or agent_2_disable or agent_3_disable:
                block_v = "X"
                block_h = "X"
            
            if self.system_state in [6,11,16] and agent_1_disable and agent_2_disable and agent_3_disable:
                print("\nFailed to enable the event\n")

            if self.system_state in [6,11,16] and not (agent_1_disable or agent_2_disable or agent_3_disable):
                print("\nSuccessfully enabled the event\n")
            
            if self.system_state in [3,8,13] and (agent_1_disable or agent_2_disable or agent_3_disable):
                print("\nSuccessfully disabled the event\n")
                
            if self.system_state in [3,8,13] and not (agent_1_disable or agent_2_disable or agent_3_disable):
                print("\nFailed to disable the event\n")
        
        
        print(
            f" +=15=+    s   +-13-+                                        +-3-+   s    +=5=+  \n"
            f" |  {a[15]} | <{block_h} {block_h} {block_h} |  {a[13]} | <--                                --> | {a[3]} | {block_h} {block_h} {block_h}> | {a[5]} |  \n"
            f" +====+        +----+    \ a                          b /    +---+        +===+  \n"
            f"                          \                            /                         \n"
            f"                          +-12-+                   +-2-+                         \n"
            f"                          |  {a[12]} |<--- c       a --->| {a[2]} |                         \n"
            f"                          +----+    \         /    +---+                         \n"
            f"                          /          \       /         \                         \n"
            f" +=16=+    s   +-14-+    / b           +-1-+          c \    +-4-+   s    +=6=+  \n"
            f" |  {a[16]} | <{block_h}-{block_h}-{block_h} |  {a[14]} | <--              | {a[1]} |             --> | {a[4]} | {block_h}-{block_h}-{block_h}> | {a[6]} |  \n"
            f" +====+        +----+                  +---+                 +---+        +===+  \n"
            f"                                         |                                       \n"
            f"                                         | b                                     \n"
            f"                                         v                                       \n"
            f"                                       +-7-+                                     \n"
            f"                                       | {a[7]} |                                     \n"
            f"                                       +---+                                     \n"
            f"                                       /   \                                     \n"
            f"                                    c /     \ a                                  \n"
            f"                                     v       v                                   \n"
            f"                                 +-8-+       +-9-+                               \n"
            f"                                 | {a[8]} |       | {a[9]} |                               \n"
            f"                                 +---+       +---+                               \n"
            f"                                   {block_v}           {block_v}                                 \n"
            f"                                 s             {block_v} s                               \n"
            f"                                   {block_v}           {block_v}                                 \n"
            f"                                   v           v                                 \n"
            f"                                +=10=+       +=11=+                              \n"
            f"                                |  {a[10]} |       |  {a[11]} |                              \n"
            f"                                +====+       +====+                              \n"
        )
        time.sleep(1)
        clear_output()
        print()
