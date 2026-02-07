import gymnasium as gym
import numpy as np
import time
from IPython.display import clear_output
from word_generator import WordGenerator

gym.register(
    id='ThreeAgentsComplexEnv-v0',
    entry_point='three_agents_complex_env:ThreeAgentsComplexEnv',
)

class ThreeAgentsComplexEnv(gym.Env):
    COMMUNICATION_COST = 10
    PENALTY = 500
    
    m_L_transitions = {
        1:{'d':2, 'b':3},
        2:{'a':4, 'c':5},
        3:{'a':5, 'c':4},
        4:{'x':6, 'b':7},
        5:{'x':7, 'b':6},
        6:{'s':8, 'b':1},
        7:{'s':9, 'b':1},
    }
    
    observer_states={
        1: "{1,2}",
        3: "{3}",
        4: "{4}",
        5: "{5}",
        6: "{6}",
        7: "{7}",
        8: "{8}",
        9: "{9}",
        -1:"{-1}"
    }
    
    observer_transitions = {
        1:{'a':4, 'b':3, 'c':5},
        3:{'a':5, 'c':4},
        4:{'x':6, 'b':7},
        5:{'x':7, 'b':6},
        6:{'s':8, 'b':1},
        7:{'s':9, 'b':1},
    }
    
    observer_bot_transitions = {
        1:{'': 1, 'a':4, 'b':3, 'c':5,     'x':-1, 's':-1},
        3:{'': 3, 'a':5, 'c':4,            'b': -1, 'x':-1, 's':-1},
        4:{'': 4, 'x':6, 'b':7,            'a': -1, 'c':-1, 's':-1},
        5:{'': 5, 'x':7, 'b':6,            'a': -1, 'c':-1, 's':-1},
        6:{'': 6, 's':8, 'b':1,            'a': -1, 'c':-1, 'x':-1},
        7:{'': 7, 's':9, 'b':1,            'a': -1, 'c':-1, 'x':-1},
        8:{'': 8,                          'a': -1, 'b':-1, 'c':-1, 'x':-1, 's':-1},
        9:{'': 9,                          'a': -1, 'b':-1, 'c':-1, 'x':-1, 's':-1},
        -1:{'': -1,                        'a': -1, 'b':-1, 'c':-1, 'x':-1, 's':-1},
    }
    
    meta_data = {'render_modes':['human'], 'string_modes':['training', 'simulation', 'stats']}
    
    def __init__(self, render_mode=None, string_mode='training', max_star=5):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-1, high=9, shape=(4,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.meta_data['render_modes'], f"Invalid render mode: {render_mode}"
        assert string_mode in self.meta_data['string_modes'], f"Invalid string mode: {string_mode}"
        
        self.render_mode = render_mode
        self.string_mode = string_mode
        
        self.word_generator = WordGenerator(max_star=max_star)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.system_state = 1
        self.agent_1_state = 1
        self.agent_2_state = 1
        self.agent_3_state = 1
        self.word_index=0
        
        
        
        if self.string_mode == 'training':
            self.word = self.word_generator.generate_training_word()+'$'
            if self.render_mode == 'human':
                print(f"\n===========New Episode=========== \nTraining word: {self.word}")
                print(f"Resulting V structure state: <{self.observer_states[self.system_state], self.observer_states[self.agent_1_state], self.observer_states[self.agent_2_state], self.observer_states[self.agent_3_state]}>")
                
            
        elif self.string_mode == 'simulation':
            self.word = self.word_generator.generate_simulation_word()+'$'
            if self.render_mode == 'human':
                print(f"\n===========New Simulation=========== \nSimulation word: {self.word}")
                self.simulate()
            
            self.curr_event = self.word[self.word_index]
            
            if self.curr_event == 'd':
                self.v_transition(['d', '', '', ''])
                self.render()
                
                self.word_index += 1
        
        self.curr_event = self.word[self.word_index]
                
        obs = (self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state)
        info  = {'word': self.word, 'curr_event': self.curr_event}
        
        return obs, info
                
    def v_transition(self, vector_label):
        if self.string_mode == 'training':
            self.system_state = self.observer_transitions[self.system_state][vector_label[0]]
        else:
            self.system_state = self.m_L_transitions[self.system_state][vector_label[0]]
        
        self.agent_1_state = self.observer_bot_transitions[self.agent_1_state][vector_label[1]]
        self.agent_2_state = self.observer_bot_transitions[self.agent_2_state][vector_label[2]]
        self.agent_3_state = self.observer_bot_transitions[self.agent_3_state][vector_label[3]]

    def get_vector_label_and_communication_cost_from_action(self, communicate, agent_id):
        vector_label = [0,0,0,0]
        
        # Determine vector label based on communication action and current event

        if agent_id==13:
            communicate_1, communicate_2 = communicate
            vector_label_1 = [1,1, communicate_1, 1]
            vector_label_2 = [1,1, communicate_2, 1]
            vector_label = np.logical_or(vector_label_1, vector_label_2).astype(int)
            communication_cost_1 = -1*communicate_1*self.COMMUNICATION_COST
            communication_cost_2 = -1*communicate_2*self.COMMUNICATION_COST
            communication_cost = communication_cost_1 , communication_cost_2
        
        else:
            if agent_id == 1:
                vector_label = [1,1,communicate[0], communicate[1]]
            if agent_id == 2:
                vector_label = [1,communicate[0],1, communicate[1]]
            if agent_id == 3:
                vector_label = [1,communicate[0], communicate[1],1]
                
            communication_cost_1 = -1*sum(communicate)*self.COMMUNICATION_COST  
            communication_cost = communication_cost_1      

        vector_label = [self.curr_event if vector_label[i]==1 else '' for i in range(4)]
            
        return vector_label, communication_cost

    def step(self, action):
        agent_id, communicate = action
        reward = 0
        terminated = False
        
        if len(communicate) !=2:
            raise ValueError("Communication action must be a list of two binary values.")
        
        if self.string_mode == 'simulation' or self.string_mode == 'stats':
            self.simulation_step(action)
        
        vector_label, communication_cost = self.get_vector_label_and_communication_cost_from_action(communicate, agent_id)
        
        # Performing the V structure transition based on the vector label, which is determined by the current event and the communication actions of the agents
        self.v_transition(vector_label)
        
        # Rendering the environment if in human mode
        if self.render_mode == 'human' and agent_id !=13:
            agents = [1,2,3]
            
            agents.remove(agent_id)
            
            receive_1 = vector_label[agents[0]]==self.curr_event
            receive_2 = vector_label[agents[1]]==self.curr_event
            
            self.render(agent_id, receive_1, agents[0], agent_id, receive_2, agents[1])
        elif self.render_mode == 'human' and agent_id == 13:
            communicate_1, communicate_2 = communicate
            self.render('1', communicate_1, '2', '3', communicate_2, '2')
        
        self.word_index += 1
        self.curr_event = self.word[self.word_index]
        
        if self.curr_event == 's':
            self.v_transition(['s', 's', 's', 's'])
            self.render()
            
            self.word_index += 1
            self.curr_event = self.word[self.word_index]
        
        # Check Termination and penalized states
        if self.system_state in [8,9] and self.agent_2_state == -1:
            terminated = True
            reward = -self.PENALTY
        elif self.curr_event == '$':
            terminated = True

        obs = (self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state)
        info = {'curr_event': self.curr_event}
        
        return np.array(obs, dtype=np.int32), (communication_cost, reward), terminated, False, info
        
    def render(self, sender_1=None, communicate_1=None, receiver_1=None, sender_2=None, communicate_2=None, receiver_2=None):
        print(f"\nEvent {self.curr_event} occurred")
        
        if self.curr_event not in ['d', 's']:
            print(f"Agent {sender_1} {'communicated' if communicate_1 else 'did not communicate'} '{self.curr_event}' to Agent {receiver_1}")
            print(f"Agent {sender_2} {'communicated' if communicate_2 else 'did not communicate'} '{self.curr_event}' to Agent {receiver_2}")

        if self.string_mode == 'training':
            print(f"Resulting V structure state: <{self.observer_states[self.system_state], self.observer_states[self.agent_1_state], self.observer_states[self.agent_2_state], self.observer_states[self.agent_3_state]}>")
        elif self.string_mode == 'simulation':
            print(f"Resulting V structure state: <{self.system_state, self.observer_states[self.agent_1_state], self.observer_states[self.agent_2_state], self.observer_states[self.agent_3_state]}>")


    def simulation_step(self, action):
        agent_id, communicate = action
        reward = 0
        
        terminated = False
        
        vector_label = [0,0,0,0]
        
        if agent_id == 1:
            vector_label = [1,1,communicate[0], communicate[1]]
        if agent_id == 2:
            vector_label = [1,communicate[0],1, communicate[1]]
        if agent_id == 3:
            vector_label = [1,communicate[0], communicate[1],1]
        
        if agent_id==13:
            communicate_1, communicate_2 = communicate
            vector_label_1 = [1,1, communicate_1, 1]
            vector_label_2 = [1,1, communicate_2, 1]
            vector_label = np.logical_or(vector_label_1, vector_label_2).astype(int)
        
        vector_label = [self.curr_event if vector_label[i]==1 else '' for i in range(4)]
        
        self.v_transition(vector_label)
        
        if agent_id == 13:
            communication_cost_1 = -1*communicate_1*self.COMMUNICATION_COST
            communication_cost_2 = -1*communicate_2*self.COMMUNICATION_COST
        else:
            communication_cost_1 = -1*sum(communicate)*self.COMMUNICATION_COST
        
        if self.render_mode == 'human' and agent_id != 13:
            agents = [1,2,3]
            
            agents.remove(agent_id)
            
            receive_1 = vector_label[agents[0]]==self.curr_event
            receive_2 = vector_label[agents[1]]==self.curr_event
            
            print(f"\nEvent: {self.curr_event} occurred.")
            print(f"Agent {agent_id} {'communicated' if receive_1 else 'did not communicate'} '{self.curr_event}' to Agent {agents[0]}")
            print(f"Agent {agent_id} {'communicated' if receive_2 else 'did not communicate'} '{self.curr_event}' to Agent {agents[1]}")
            self.render()
    

        if self.render_mode == 'human' and agent_id == 13:
            print(f"\nEvent 'x' occurred")
            print(f"Agent 1 {'communicated' if communicate_1 else 'did not communicate'} 'x' to Agent 2")
            print(f"Agent 3 {'communicated' if communicate_2 else 'did not communicate'} 'x' to Agent 2")
            self.render()
    
        self.word_index += 1
        self.curr_event = self.word[self.word_index]
        
        if self.curr_event == 'd':
            self.v_transition(['d', '', '', ''])
            
            if self.render_mode == 'human':
                print("Event d occurred")
                self.render()
            
            self.word_index += 1
            self.curr_event = self.word[self.word_index]
            
        
        elif self.curr_event == 's':
            
            agent_1_disable = self.agent_1_state == 6
            agent_2_disable = self.agent_2_state == 6
            agent_3_disable = self.agent_3_state == 6
            
            if not (agent_1_disable or agent_2_disable or agent_3_disable):
                self.v_transition(['s', 's', 's', 's'])
                
            
            if self.render_mode == 'human':
                print("Event s occurred")
                self.render()
                self.simulate(agent_1_disable=agent_1_disable, agent_2_disable=agent_2_disable, agent_3_disable=agent_3_disable)
            
            self.word_index += 1
            self.curr_event = self.word[self.word_index]
            if self.system_state == 9 and not (agent_1_disable or agent_2_disable or agent_3_disable):
                simulation_result = True
            
            if self.system_state == 8 and (agent_1_disable or agent_2_disable or agent_3_disable):
                simulation_result = True
        
            if self.curr_event == '$':
                terminated = True
                if not simulation_result:
                    reward = -self.PENALTY

        obs = (self.system_state, self.agent_1_state, self.agent_2_state, self.agent_3_state)
        info = {'curr_event': self.curr_event}
        
        if agent_id == 13:
            communication_cost = communication_cost_1 , communication_cost_2
        else:
            communication_cost = communication_cost_1
        
        return np.array(obs, dtype=np.int32), (communication_cost, reward), terminated, False, info   
    
    def simulate(self, agent_1_disable=False, agent_2_disable=False, agent_3_disable=False):
        a = [" " for _ in range(10)]
        a[self.system_state] = "#"
        
        block = "|"
        
        if self.curr_event == 's':
            block = "X"
            
            if self.system_state == 9 and not (agent_1_disable or agent_2_disable or agent_3_disable):
                print("\nSuccessfully enabled s")
            
            if self.system_state == 9 and (agent_1_disable or agent_2_disable or agent_3_disable):
                print("\nFailed to enable s")
            
            if self.system_state == 8 and (agent_1_disable or agent_2_disable or agent_3_disable):
                print("\nSuccessfully disabled s")
                
            if self.system_state == 8 and not (agent_1_disable or agent_2_disable or agent_3_disable):
                print("\nFailed to disable s")
        
        print(
             "\n              start                     \n"
             "                  |                         \n"
             "                  V                         \n"
             "                +-1-+                       \n"
            f"    ----------->| {a[1]} |<-----------           \n"
             "    |           +---+           |               \n"
             "    |         /       \\         |               \n"
             "    |      d /         \\ b      |               \n"
             "    |       /           \\       |               \n"
             "    |      V             V      |               \n"
             "    |   +-2-+           +-3-+   |               \n"
            f"    |   | {a[2]} |           | {a[3]} |   |               \n"
             "    |   +---+ --     -- +---+   |               \n"
             "    |     |     \\   /     |     |               \n"
             "    |   a |      \\ /      | a   |               \n"
             " b  |     |       c       |     |  b            \n"
             "    |     V      / \\      V     |               \n"
             "    |   +-4-+ <--   --> +-5-+   |               \n"
            f"    |   | {a[4]} |           | {a[5]} |   |               \n"
             "    |   +---+ --     -- +---+   |               \n"
             "    |     |     \\   /     |     |               \n"
             "    |   x |      \\ /      | x   |               \n"
             "    |     |       b       |     |               \n"
             "    |     V      / \\      V     |               \n"
             "    |   +-6-+ <--   --> +-7-+   |               \n"
            f"    ----| {a[6]} |           | {a[7]} |----               \n"
             "        +---+           +---+                   \n"
            f"          {block}               {block}                     \n"
            f"        s                 {block} s                   \n"
            f"          {block}               {block}                     \n"
             "          V               V                     \n"
             "        +-8-+           +-9-+                   \n"
            f"        | {a[8]} |           | {a[9]} |                   \n"
             "        +---+           +---+                   \n"
        )
        time.sleep(1)
        clear_output()
        print()