'''
  mab_agent.py
  
  Agent specifications implementing Action Selection Rules.
'''

import numpy as np
import random
from math import e, sqrt

# ----------------------------------------------------------------
# MAB Agent Superclasses
# ----------------------------------------------------------------

# Returns a list of action values
def compute_action_value(history):
    list_of_av = []
    for x in np.array(history):
        list_of_av.append((x[1]/x[0]))
    return list_of_av

def compute_current_trial_num(current_history):
    sum = 0
    for i in range(len(current_history)):
        sum += current_history[i][0]
    # Sum - 2 because we initially fabricate two trials to prevent a divide by zero error when computing the action value
    return (sum - 2)

# Returns an empty 2D array of history
def get_empty_history(K):
    history = []
    for r in range(K):
        row = []
        for c in range(2):
            if c == 0:
                row.append(2)
            else:
                row.append(1)
        history.append(row)
    return history


class MAB_Agent:
    '''
    MAB Agent superclass designed to abstract common components
    between individual bandit players (below)
    '''   
    def __init__ (self, K):
        self.K = K
        self.epsilon = None
        self.current_action = None
        self.current_reward = None
        self.trial = 0
        self.T = None
        # History is a 2D list such that each index in the list represents an arm and each element is 2 element list for that arm's
        # total number of pulls and total number of rewards
        self.history = get_empty_history(self.K)

        
    def give_feedback (self, a_t, r_t):
        '''
        Provides the action a_t and reward r_t chosen and received
        in the most recent trial, allowing the agent to update its
        history
        [!] Called by the simulations after your agent's choose
            method is called
        '''
        self.trial += 1
        self.current_action = a_t
        self.current_reward = r_t
        self.history[a_t][0] += 1
        self.history[a_t][1] += r_t

    
    def clear_history(self):
        '''
        IMPORTANT: Resets your agent's history between simulations.
        No information is allowed to transfer between each of the N
        repetitions
        [!] Called by the simulations after a Monte Carlo repetition
        '''
        self.trial = 0
        self.history = get_empty_history(self.K)


# ----------------------------------------------------------------
# MAB Agent Subclasses
# ----------------------------------------------------------------

class Greedy_Agent(MAB_Agent):
    '''
    Greedy bandit player that, at every trial, selects the
    arm with the presently-highest sampled Q value
    '''
    def __init__ (self, K):
        MAB_Agent.__init__(self, K)
        
    def choose (self, *args):
        action = np.array(compute_action_value(self.history))
        max = np.argwhere(action == np.amax(action)).flatten().tolist()
        if len(max) == 1:
            choice = (max[0])
        else:
            choice = (random.choice(max))
        return choice


class Epsilon_Greedy_Agent(MAB_Agent):
    '''
    Exploratory bandit player that makes the greedy choice with
    probability 1-epsilon, and chooses randomly with probability
    epsilon
    '''  
    def __init__ (self, K, epsilon):
        MAB_Agent.__init__(self, K)
        self.epsilon = epsilon
            
    def choose (self, *args):
        if random.random() <= 1 - self.epsilon:
            action = np.array(compute_action_value(self.history))
            max = np.argwhere(action == np.amax(action)).flatten().tolist()
            if len(max) == 1:
                choice = (max[0])
            else:
                choice = (random.choice(max))
            return choice
        return np.random.choice(list(range(self.K)))
        


class Epsilon_First_Agent(MAB_Agent):
    '''
    Exploratory bandit player that takes the first epsilon*T
    trials to randomly explore, and thereafter chooses greedily
    '''    
    def __init__ (self, K, epsilon, T):
        MAB_Agent.__init__(self, K)
        self.epsilon = epsilon
        self.T = T
        
    def choose (self, *args):
        if self.trial < self.T * self.epsilon:
            return np.random.choice(list(range(self.K)))
        action = np.array(compute_action_value(self.history))
        max = np.argwhere(action == np.amax(action)).flatten().tolist()
        if len(max) == 1:
            choice = (max[0])
        else:
            choice = (random.choice(max))
        return choice
         


class Epsilon_Decreasing_Agent(MAB_Agent):
    '''
    Exploratory bandit player that acts like epsilon-greedy but
    with a decreasing value of epsilon over time
    '''  
    def __init__ (self, K, T, i):
        MAB_Agent.__init__(self, K)
        self.epsilon = 0.15
        self.T = T
        self.function = i
        
    def choose (self, *args):
        if self.function == 0:
            # Function #1
            self.epsilon -= 0.001
        elif self.function == 1:
            # Function #2
            self.epsilon = (e ** -(self.trial + 2))
        else:
            # Function #3
            self.epsilon = 1/((self.trial + 45) ** 0.5)
        if self.epsilon < 0:
                self.epsilon = 0
        if random.random() <= 1 - self.epsilon:
            action = np.array(compute_action_value(self.history))
            max = np.argwhere(action == np.amax(action)).flatten().tolist()
            if len(max) == 1:
                choice = (max[0])
            else:
                choice = (random.choice(max))
            return choice
        return np.random.choice(list(range(self.K)))


class TS_Agent(MAB_Agent):
    '''
    Thompson Sampling bandit player that self-adjusts exploration
    vs. exploitation by sampling arm qualities from successes
    summarized by a corresponding beta distribution
    '''
    
    def __init__ (self, K):
        MAB_Agent.__init__(self, K)
    
    def choose (self, *args):
        # TODO: Currently makes a random choice -- change!
        # Hint: you'll use your agent's history and observed
        # feedback here
        return np.random.choice(list(range(self.K)))
    
    
class Custom_Agent(MAB_Agent):
    '''
    Custom agent that manages the explore vs. exploit dilemma via
    your own strategy, or by implementing a strategy you discovered
    that is not amongst those above!
    '''
    
    def __init__ (self, K, T, c, epsilon):
        MAB_Agent.__init__(self, K)
        self.T = T
        self.K = K
        self.c = c
        self.epsilon = epsilon
    
    def choose (self, *args):
        # TODO: Currently makes a random choice -- change!
        # Hint: you'll use your agent's history and observed
        # feedback here
        if self.trial < self.T * self.epsilon:
            return np.random.choice(list(range(self.K)))
        action = np.array(compute_action_value(self.history))
        ucb_action = []
        for i, a_v in enumerate(action):
            numerator = np.log(self.trial + 1)
            denominator = self.history[i][0]
            sqr = sqrt(numerator/denominator)
            ucb = a_v + (self.c * sqr)
            ucb_action.append(ucb)
        action = np.array(ucb_action)
        max = np.argwhere(action == np.amax(action)).flatten().tolist()
        if len(max) == 1:
            choice = (max[0])
        else:                
            choice = (random.choice(max))
        return choice

