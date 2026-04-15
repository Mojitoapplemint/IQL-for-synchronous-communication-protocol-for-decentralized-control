import gymnasium as gym
import numpy as np
import time
from word_generator import WordGenerator
from IPython.display import clear_output
import pandas as pd

gym.register(
    id='DistributiveEnv-v1',
    entry_point='distributive_env:DistributiveEnv',
)

class DistributiveEnv(gym.Env):
    COMMUNICATION_COST = 2
    D_PENALTY = 10
    E_PENALTY = 0
    
    D_PEN_STATES = {6}
    
    E_PEN_STATES = {8,9}
    
    STATES_DISABLE_SIGMA = {3}
    
    m_L_transitions = {
        0:{'a':1, 'b':2},
        1:{'b':3, 'd':4},
        2:{'a':5},
        3:{'s':6},
        4:{'b':7},
        5:{'s':8},
        7:{'s':9},
    }
    
    m_L_bot_transitions = {
        0:{'':0, 'a':1, 'b':2, 'd':-1, 's':-1},
        1:{'':1, 'b':3, 'd':4, 's':-1, 'a':-1},
        2:{'':2, 'a':5,       's':-1, 'b':-1, 'd':-1},
        3:{'':3, 's':6,       'a':-1, 'b':-1, 'd':-1},
        4:{'':4, 'b':7,       's':-1, 'a':-1, 'd':-1},
        5:{'':5, 's':8,       'a':-1, 'b':-1, 'd':-1},
        6:{'':6,              's':-1, 'a':-1, 'b':-1, 'd':-1},
        7:{'':7, 's':9,       'a':-1, 'b':-1, 'd':-1},
        8:{'':8,              's':-1, 'a':-1, 'b':-1, 'd':-1},
        9:{'':9,              's':-1, 'a':-1, 'b':-1, 'd':-1},
        -1:{'':-1,              's':-1, 'a':-1, 'b':-1, 'd':-1},
    }
    
    meta_data = {'render_modes':['human'], 'string_modes':['training', 'simulation']}
    
    def __init__(self, render_mode=None, string_mode='training', max_star=3):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=9, shape=(3,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.meta_data['render_modes'], f"Invalid render mode: {render_mode}"
        assert string_mode in self.meta_data['string_modes'], f"Invalid string mode: {string_mode}"
        
        self.render_mode = render_mode
        self.string_mode = string_mode
        
        self.word_generator = WordGenerator(max_star=max_star)
        
        self.simulation_words = pd.read_csv('check_distributive_observability/simulation_words.csv')['word'].tolist()
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
        
        if self.curr_event not in ['s']:
            print(f"Agent {sender_1} {'communicated' if communicate_1 else 'did not communicate'} '{self.curr_event}' to Agent {receiver_1}")


        if self.string_mode == 'training':
            print(f"Resulting V structure state: <{self.system_state}, {self.agent_1_state}, {self.agent_2_state}>")
        elif self.string_mode == 'simulation':
            print(f"Resulting V structure state: <{self.system_state}, {self.agent_1_state}, {self.agent_2_state}>")

    
    def simulate(self, agent_1_disable=False, agent_2_disable=False):
        a = [" " for _ in range(10)]
        a[self.system_state] = "#"
        
        d_block = ":"
        e_block = "|"

        
        if self.curr_event == 's':
            if agent_2_disable:
                print("Agent 2 disabled 's'")
                e_block = "X"
                d_block = "X"
            
            if agent_1_disable:
                print("Agent 1 disabled 's'")
                e_block = "X"
                d_block = "X"
                 
            if self.system_state in self.E_PEN_STATES and not (agent_2_disable or agent_1_disable):
                print("\nSuccessfully enabled s")
            
            if self.system_state in self.D_PEN_STATES and (agent_2_disable or agent_1_disable):
                print("\nFailed to enable s")
            
            if self.system_state == self.STATES_DISABLE_SIGMA and (agent_2_disable or agent_1_disable):
                print("\nSuccessfully disabled s")
                
            if self.system_state == self.D_PEN_STATES and not (agent_2_disable or agent_1_disable):
                print("\nFailed to disable s")
        
        print(
             "                                                        \n"
             "                             +-00-+                     \n"
            f"                             | {a[0]}  |                     \n"
             "                             +----+                     \n"
             "                          a /      \\ b                 \n"
             "                           v        v                   \n"
             "                     +-01-+          +-02-+             \n"
            f"                     | {a[1]}  |          | {a[2]}  |             \n"
             "                     +----+          +----+             \n"
             "                    /     |             |               \n"
             "                 b /      | d           | a             \n"
             "                  v       v             v               \n"
             "              +-03-+    +-04-+       +-05-+             \n"
            f"              | {a[3]}  |    | {a[4]}  |       | {a[5]}  |             \n"
             "              +----+    +----+       +----+             \n"
            f"                 {d_block}        |             {e_block}               \n"
            f"               s {d_block}        | b           {e_block} s             \n"
             "                 v        v             v               \n"
             "              +-06-+    +-07-+       +-08-+             \n"
            f"              | {a[6]}  |    | {a[7]}  |       | {a[8]}  |             \n"
             "              +----+    +----+       +----+             \n"
            f"                          {e_block}                             \n"
            f"                          {e_block} s                           \n"
             "                          v                             \n"
             "                        +-09-+                          \n"
            f"                        | {a[9]}  |                          \n"
             "                        +----+                          \n"
        )

        time.sleep(1)
        clear_output()
        print()