import gymnasium as gym
import numpy as np
import time
from word_generator import WordGenerator
from IPython.display import clear_output
import pandas as pd

gym.register(
    id='ThreeAgentsLSEnv-v0',
    entry_point='three_agents_ls_env:ThreeAgentsLSEnv',
)

class ThreeAgentsLSEnv(gym.Env):
    COMMUNICATION_COST = 2.5
    EXPENSIVE_COMMUNICATION_COST = 100
    EXPENSIVE_COMMUNICATION = ['x', 'y']
    D_PENALTY = 400
    E_PENALTY = 0
    
    D_PEN_STATES = {10,13}
    
    E_PEN_STATES = {11,12}
    
    STATES_DISABLE_SIGMA = {6,9}
    
    m_L_transitions = {
        1:{'c':2, 'a':3},
        2:{'a':4},
        3:{'c':5},
        4:{'x':6, 'y':7},
        5:{'x':8, 'y':9},
        6:{'s':10},        
        7:{'s':11},
        8:{'s':12},
        9:{'s':13},
    }
    
    m_L_bot_transitions = {
        1: {'':1, 'c':2, 'a':3          , 'x':-1, 'y':-1, 's':-1},
        2: {'':2, 'a':4                 , 'c':-1, 'x':-1, 'y':-1, 's':-1},
        3: {'':3, 'c':5                 , 'a':-1, 'x':-1, 'y':-1, 's':-1},
        4: {'':4, 'x':6, 'y':7          , 'a':-1, 'c':-1, 's':-1},
        5: {'':5, 'x':8, 'y':9          , 'a':-1, 'c':-1, 's':-1},
        6: {'':6, 's':10                , 'c':-1, 'a':-1, 'x':-1, 'y':-1},        
        7: {'':7, 's':11                , 'c':-1, 'a':-1, 'x':-1, 'y':-1},
        8: {'':8, 's':12                , 'c':-1, 'a':-1, 'x':-1, 'y':-1},
        9: {'':9, 's':13                , 'c':-1, 'a':-1, 'x':-1, 'y':-1},
        10:{'':10, 's':-1 , 'c':-1, 'a':-1, 'x':-1, 'y':-1},
        11:{'':11, 's':-1 , 'c':-1, 'a':-1, 'x':-1, 'y':-1},
        12:{'':12, 's':-1 , 'c':-1, 'a':-1, 'x':-1, 'y':-1},
        13:{'':13, 's':-1 , 'c':-1, 'a':-1, 'x':-1, 'y':-1},
        -1:{'':-1, 's':-1 , 'c':-1, 'a':-1, 'x':-1, 'y':-1},
    }
    
    meta_data = {'render_modes':['human'], 'string_modes':['training', 'simulation', 'stats']}
    
    def __init__(self, render_mode=None, string_mode='training', max_star=5):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-1, high=15, shape=(4,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.meta_data['render_modes'], f"Invalid render mode: {render_mode}"
        assert string_mode in self.meta_data['string_modes'], f"Invalid string mode: {string_mode}"
        
        self.render_mode = render_mode
        self.string_mode = string_mode
        
        self.word_generator = WordGenerator(max_star=max_star)
        
        self.simulation_words = pd.read_csv('three_agents_long_short/words_for_stats.csv')['word'].tolist()
        self.simulation_words_index = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state = 1, 1, 1, 1
        
        self.word_index=0
        
        if self.string_mode == 'training':
            self.word = self.word_generator.generate_training_word()+'$'
            if self.render_mode == 'human':
                print(f"\n===========New Episode=========== \nTraining word: {self.word}")
                print(f"Resulting V structure state: <{[self.system_state], [self.agent_1_state], [self.agent_2_state], [self.agent_3_state]}>")
                
            
        elif self.string_mode == 'simulation':
            self.word = self.simulation_words[self.simulation_words_index] + '$'
            self.simulation_words_index += 1
            self.curr_event = self.word[self.word_index]
                        
            if self.render_mode == 'human':
                print(f"\n===========New Simulation=========== \nSimulation word: {self.word}")
                self.simulate()
        
        self.curr_event = self.word[self.word_index]
        
        # while self.curr_event == 'd':
        #     self.v_transition(['d', '', '', ''])
        #     if self.render_mode == 'human':
        #         self.render()
        #         self.simulate()
            
        #     self.word_index += 1
        #     self.curr_event = self.word[self.word_index]
                
        v_state = (self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state)
        info  = {'word': self.word, 'curr_event': self.curr_event}
        
        return v_state, info
                
    def v_transition(self, vector_label):
        if self.string_mode == 'training':
            self.system_state = self.m_L_transitions[self.system_state][vector_label[0]]
        else:
            self.system_state = self.m_L_transitions[self.system_state][vector_label[0]]
        
        self.agent_1_state = self.m_L_bot_transitions[self.agent_1_state][vector_label[1]]
        self.agent_2_state = self.m_L_bot_transitions[self.agent_2_state][vector_label[2]]
        self.agent_3_state = self.m_L_bot_transitions[self.agent_3_state][vector_label[3]]

    def get_vector_label_and_communication_cost_from_action(self, communicate, agent_id):
        vector_label = [0,0,0,0]
        
        if agent_id == 1:
            vector_label = [1,1,communicate[0], communicate[1]]
        if agent_id == 2:
            vector_label = [1,communicate[0],1, communicate[1]]
        if agent_id == 3:
            vector_label = [1,communicate[0], communicate[1],1]
        
        if self.curr_event in self.EXPENSIVE_COMMUNICATION:
            communication_cost_1 = -1*sum(communicate)*self.EXPENSIVE_COMMUNICATION_COST
        else:    
           communication_cost_1 = -1*sum(communicate)*self.COMMUNICATION_COST 
            
        communication_cost = communication_cost_1      

        vector_label = [self.curr_event if vector_label[i]==1 else '' for i in range(4)]
            
        return vector_label, communication_cost

    def step(self, action):
        agent_id, communicate = action
        reward = 0
        terminated = False
        simulation_result = False
        
        if len(communicate) !=2:
            raise ValueError("Communication action must be a list of two binary values.")
        
        vector_label, communication_cost = self.get_vector_label_and_communication_cost_from_action(communicate, agent_id)
        
        # Performing the V structure transition based on the vector label, which is determined by the current event and the communication actions of the agents
        self.v_transition(vector_label)
        
        # Rendering the environment if in human mode
        if self.render_mode == 'human':
            agents = [1,2,3]
            
            agents.remove(agent_id)
            
            receive_1 = vector_label[agents[0]]==self.curr_event
            receive_2 = vector_label[agents[1]]==self.curr_event
            
            self.render(agent_id, receive_1, agents[0], agent_id, receive_2, agents[1])
            
            if self.string_mode == 'simulation':
                self.simulate()
        
        # Updating the current event to the next event in the word
        self.word_index += 1
        self.curr_event = self.word[self.word_index]
        
        # State transition for completely unobservable event 'd' (Only applied when simulation/stats mode)
        # while self.curr_event == 'd':
        #     self.v_transition(['d', '', '', ''])
            
        #     if self.render_mode == 'human':
        #         self.render()
            
        #     self.word_index += 1
        #     self.curr_event = self.word[self.word_index]
        
        # State transition for 's' for training mode
        if self.curr_event == 's' and self.string_mode == 'training':
            self.v_transition(['s', 's', 's', 's'])
            if self.render_mode == 'human':
                self.render()
            
            # Updating current event to '$' indicating end of the word
            self.word_index += 1
            self.curr_event = self.word[self.word_index]
            
        # State transition for 's' for simulation or stats mode
        if self.curr_event == 's' and (self.string_mode == 'simulation' or self.string_mode == 'stats'):
            
            # If agent 2 think it is in state where s must be disabled, then it disables it
            agent_2_disable = self.agent_2_state in self.STATES_DISABLE_SIGMA
            
            if not (agent_2_disable):
                self.v_transition(['s', 's', 's', 's'])
                
            if self.render_mode == 'human':
                self.render()
                self.simulate(agent_2_disable=agent_2_disable)
            
            
            
            # Check simulation results based on system state and agent 2's disable status
            if (self.system_state in self.E_PEN_STATES) and not (agent_2_disable):
                simulation_result = True

            if (self.system_state in self.STATES_DISABLE_SIGMA) and (agent_2_disable):
                simulation_result = True
                
            # Updating current event to '$' indicating end of the word
            self.word_index += 1
            self.curr_event = self.word[self.word_index]
        
        
        if (self.system_state in self.D_PEN_STATES) and self.agent_2_state ==-1:
            terminated = True
            reward = -self.D_PENALTY
        if (self.system_state in self.E_PEN_STATES) and self.agent_2_state ==-1:
            terminated = True
            reward = -self.E_PENALTY
        elif self.curr_event == '$':
            terminated = True
        

        v_state = (self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state)
        info = {'curr_event': self.curr_event}
        
        if self.string_mode == 'simulation' or self.string_mode == 'stats':
            truncated = simulation_result
        else:
            truncated = False
        
        return np.array(v_state, dtype=np.int32), (communication_cost, reward), terminated, truncated, info
        
    def render(self, sender_1=None, communicate_1=None, receiver_1=None, sender_2=None, communicate_2=None, receiver_2=None):
        print(f"\nEvent {self.curr_event} occurred")
        
        if self.curr_event not in ['d', 's']:
            print(f"Agent {sender_1} {'communicated' if communicate_1 else 'did not communicate'} '{self.curr_event}' to Agent {receiver_1}")
            print(f"Agent {sender_2} {'communicated' if communicate_2 else 'did not communicate'} '{self.curr_event}' to Agent {receiver_2}")


        if self.string_mode == 'training':
            print(f"Resulting V structure state: <{self.system_state}, {self.agent_1_state}, {self.agent_2_state}, {self.agent_3_state}>")
        elif self.string_mode == 'simulation':
            print(f"Resulting V structure state: <{self.system_state}, {self.agent_1_state}, {self.agent_2_state}, {self.agent_3_state}>")

    
    def simulate(self, agent_2_disable=False):
        a = [" " for _ in range(14)]
        a[self.system_state] = "#"
        
        e_block = "|"
        d_block = ":"
        
        if self.curr_event == 's':
            if agent_2_disable:
                print("Agent 2 disabled 's'")
                e_block = "X"
                d_block = "X"
            
            if self.system_state in self.E_PEN_STATES and not (agent_2_disable):
                print("\nSuccessfully enabled s")
            
            if self.system_state in self.D_PEN_STATES and (agent_2_disable):
                print("\nFailed to enable s")
            
            if self.system_state == self.STATES_DISABLE_SIGMA and (agent_2_disable):
                print("\nSuccessfully disabled s")
                
            if self.system_state == self.D_PEN_STATES and not (agent_2_disable):
                print("\nFailed to disable s")
        
        print(
         "                               +-01-+                           \n"
        f"                      ---------|  {a[1]} |---------                  \n"
         "                      |        +----+        |                  \n"
         "                   c  |                      |   a              \n"
         "                      |                      |                  \n"
         "                      v                      v                  \n"
         "                   +-02-+                 +-03-+                \n"
        f"                   |  {a[2]} |                 |  {a[3]} |                \n"
         "                   +----+                 +----+                \n"
        f"                      |                      |                  \n"
        f"                    a |                      | c                \n"
         "                      v                      v                  \n"
         "                   +-04-+                 +-05-+                \n"
        f"                   |  {a[4]} |                 |  {a[5]} |                \n"
         "                   +----+                 +----+                \n"
        f"                    /   \\                 /   \\               \n"
        f"                 x /     \\ y           x /     \\ y            \n"
         "                  v       v              v       v              \n"
         "               +-06-+   +-07-+        +-08-+   +-09-+           \n"
        f"               |  {a[6]} |   |  {a[7]} |        |  {a[8]} |   |  {a[9]} |           \n"
         "               +----+   +----+        +----+   +----+           \n"
        f"                  {d_block}       {e_block}              {e_block}       {d_block}              \n"
        f"                s {d_block}       {e_block} s          s {e_block}       {d_block} s            \n"
         "                  v       v              v       v              \n"
         "               +-10-+   +-11-+        +-12-+   +-13-+           \n"
        f"               |  {a[10]} |   |  {a[11]} |        |  {a[12]} |   |  {a[13]} |           \n"
         "               +----+   +----+        +----+   +----+           \n")

        time.sleep(1)
        clear_output()
        print()