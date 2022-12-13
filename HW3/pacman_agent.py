'''
Pacman Agent employing a PacNet trained in another module to
navigate perilous ghostly pellet mazes.
'''

import time
import random
import numpy as np
import torch
from os.path import exists
from torch import nn
from pathfinder import *
from queue import Queue
from constants import *
from reinforcement_trainer import *
from maze_problem import *

class PacmanAgent:
    '''
    Deep learning Pacman agent that employs PacNet DQNs.
    '''

    def __init__(self, maze):
        """
        Initializes the PacmanAgent with any attributes needed to make decisions;
        for the deep-learning implementation, this includes initializing the
        policy DQN (+ target DQN, ReplayMemory, and optimizer if training) and
        any other 
        :maze: The maze on which this agent is to operate. Must be the same maze
        structure as the one on which this agent's model was trained.
        """
        # [!] TODO!
        self.maze = maze
        self.policy_net = PacNet(maze)

        # Load possible previously existing weights for policy network
        #self.policy_net.load_state_dict(torch.load(Constants.PARAM_PATH))
        if Constants.TRAINING:
            self.target_net = PacNet(maze).to(Constants.DEVICE)
            self.replay_mem = ReplayMemory(Constants.MEM_SIZE)
            self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        
        self.steps_done = 0

        if exists(Constants.MEM_PATH):
            self.replay_mem.load()
        if exists(Constants.PARAM_PATH):
            self.policy_net.load_state_dict(torch.load(Constants.PARAM_PATH))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from the options in Constants.MOVES based on the agent's
        perception (the current maze) and legal actions available. If training,
        must manage the explore vs. exploit dilemma through some form of ASR.
        :perception: The current maze state in which to act
        :legal_actions: Map of legal actions to their next agent states
        :return: Action choice from the set of legal_actions
        """
        # [!] TODO!
        legal_actions = dict(legal_actions)
        with torch.no_grad():
            Q_values = self.policy_net(self.get_nn_input(perception))
            Q_values = Q_values[0].numpy().tolist()
            removed_indices = self.get_removed_indices(list(legal_actions.keys()))
            #Credit for removing multiple elements at particular indices at same time: https://stackoverflow.com/a/11303241
            removed_Q_values = [i for j, i in enumerate(Q_values) if j not in removed_indices]
        if Constants.TRAINING:
            p = random.random()
            if p < (1 - Constants.EPS_GREEDY):
                return Constants.MOVES[Q_values.index(max(removed_Q_values))]
            else:
                return random.choice(list(legal_actions.keys()))
        return Constants.MOVES[Q_values.index(max(removed_Q_values))]
    
    # Returns a list of indices to be removed from the list of Q_values returned from PacNet
    @staticmethod
    def get_removed_indices(legal_action_list):
        list = []      
        for i in range(len(Constants.MOVES)):
            if not (Constants.MOVES[i] in legal_action_list):
                list.append(i)
        return list

    # Returns a vectorized maze-state for the neural network input
    @staticmethod
    def get_nn_input(maze_state):
        return torch.from_numpy(ReplayMemory.vectorize_maze(maze_state)).float().unsqueeze(0)
    
    def get_reward(self, state, action, next_state):
        '''
        The reward function that determines the numerical desirability of the
        given transition from state -> next_state with the chosen action.
        :state: state at which the transition begun
        :action: the action the agent chose from state
        :next_state: the state at which the agent began its next turn
        :returns: R(s, a, s') for the given transition
        '''
        # [!] TODO!
        s = MazeProblem(state)
        s_prime = MazeProblem(next_state)
        if s_prime.is_terminal:
            if s_prime.get_win_state is not None:
                return 500
            if s_prime.get_death_state is not None:
                return -500
            if s_prime.get_timeout_state is not None:
                return -400
        if len(s.get_pellets) > len(s_prime.get_pellets):
            return 300
        return 50
    
    def give_transition(self, state, action, next_state, is_terminal):
        '''
        Called by the Environment after both Pacman and ghosts have moved on a
        given turn, supplying the transition that was observed, which can then
        be added to the training agent's memory and the model optimized. Also
        responsible for periodically updating the target network.
        [!] If not training, this method should do nothing.
        :state: state at which the transition begun
        :action: the action the agent chose from state
        :next_state: the state at which the agent began its next turn
        :is_terminal: whether or not next_state is a terminal state
        '''
        # [!] TODO!
        if not Constants.TRAINING:
            return
        reward = self.get_reward(state, action, next_state)
        self.replay_mem.push(state, action, next_state, reward, is_terminal)
        self.steps_done += 1
        self.optimize_model()

        if self.steps_done%Constants.TARGET_UPDATE==0 and self.steps_done>1:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.give_terminal()     
    
    def give_terminal(self):
        '''
        Called by the Environment upon reaching any of the terminal states:
          - Winning (eating all of the pellets)
          - Dying (getting eaten by a ghost)
          - Timing out (taking more than Constants.MAX_MOVES number of turns)
        Useful for cleaning up fields, saving weights and memories to disk if
        desired, etc.
        [!] If not training, this method should do nothing.
        '''
        # [!] TODO!
        if Constants.TRAINING:
            self.replay_mem.save()
    
    def optimize_model(self):
        '''
        Primary workhorse for training the policy DQN. Samples a mini-batch of
        episodes from the ReplayMemory and then takes a step of the optimizer
        to train the DQN weights.
        [!] If not training OR fewer episodes than Constants.BATCH_SIZE have
        been recorded, this method should do nothing.
        '''
        # [!] TODO!
        if (not Constants.TRAINING) or len(self.replay_mem) < Constants.BATCH_SIZE:
            return

        transitions = self.replay_mem.sample(Constants.BATCH_SIZE)
        batch = Episode(*zip(*transitions))

        non_final_mask = torch.tensor([not b for b in batch.is_terminal], dtype=torch.bool)
        non_final_next_states = torch.tensor(np.array([ReplayMemory.vectorize_maze(batch.next_state[i]) for i in range(len(batch.next_state)) if not batch.is_terminal[i]])).float()
        
        state_batch = torch.tensor(np.array([ReplayMemory.vectorize_maze(s) for s in batch.state])).float()
        action_batch = torch.tensor([[ReplayMemory.move_vec_to_index(ReplayMemory.vectorize_move(a))] for a in batch.action])
        reward_batch = torch.tensor(batch.reward).float()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(Constants.BATCH_SIZE, device=Constants.DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * Constants.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    

        
