from gymnasium.envs.toy_text.taxi import TaxiEnv
from gymnasium.wrappers import TimeLimit
import numpy as np
import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space



class ExtendedTaxiEnv(TaxiEnv):

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        
        ####### ----------------  original ------------------##################
        define_distribution(self, difficulty_level = options['difficulty_level'])
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}
        ####### ----------------  replacment ------------------##################
        # low, high = utils.maybe_parse_reset_bounds(
        #     options, -0.05, 0.05  # default low
        # )  # default high
        
     
    


def define_distribution(self, difficulty_level = 'normal'):
    
    self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

    num_states = 500
    num_rows = 5
    num_columns = 5
    self.initial_state_distrib = np.zeros(num_states)
    
    ####------------------------------ original init -------------------------------####
    if difficulty_level == "normal":
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx: # if passanger is not in the taxi, and passanger/dest in different locations
                            self.initial_state_distrib[state] += 1
                        
        self.initial_state_distrib /= self.initial_state_distrib.sum()

    #### -------------------------passanger in the taxi-------------------------####
    elif difficulty_level == "passanger_in_taxi":
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx == 4:
                            self.initial_state_distrib[state] += 1
                        
        self.initial_state_distrib /= self.initial_state_distrib.sum()


    #### -------------------------passanger and taxi in the same location-------------------------####
    elif difficulty_level == "passanger_and_taxi_in_same_place": 
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx and (row,col) == locs[pass_idx]:
                            self.initial_state_distrib[state] += 1
                        
        self.initial_state_distrib /= self.initial_state_distrib.sum()

    #### -------------------------passanger and taxi in the same location, taxi close to destenation-------------------------####
    else: 
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx == 4 and pass_idx != dest_idx:
                            if dest_idx == 0: # (0, 0)
                                if row < 2:
                                    if col < 2:
                                        self.initial_state_distrib[state] += 1
                            if dest_idx == 1: #  (0, 4)
                                if row < 2:
                                    if col > 2:
                                        self.initial_state_distrib[state] += 1
                            if dest_idx == 2: # (4, 0)
                                if row > 2:
                                    if col < 2:
                                        self.initial_state_distrib[state] += 1
                            if dest_idx == 3: # (4, 3)
                                if row > 2:
                                    if col > 2:
                                        self.initial_state_distrib[state] += 1

                                

                            
                       
        self.initial_state_distrib /= self.initial_state_distrib.sum()



class DetermenisticTaxiEnv(TaxiEnv):

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, state = None):
        
        self.s = state
        self.lastaction = None
        self.taxi_orientation = 0

        
        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}