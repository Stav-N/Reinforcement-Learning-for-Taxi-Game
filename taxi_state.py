import numpy as np
from taxi_wrapper import ExtendedTaxiEnv, DetermenisticTaxiEnv
from gymnasium.wrappers import TimeLimit
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



class TaxiState():
    def __init__(self, state = None):
        """ class to represent state"""
        options = {}
        options['difficulty_level'] = "normal"
        if state:
            self.env = TimeLimit(env = DetermenisticTaxiEnv(), max_episode_steps = 200)
            self.state, self.info = self.env.reset(state = state, options = options)
        else: 
            self.env = TimeLimit(env = ExtendedTaxiEnv(), max_episode_steps = 200)
            self.state, self.info = self.env.reset(options = options)

    def is_goal(self):
        if self.state in [0,85,410,495]:
            return True
        else:
            return False
        
    def get_state_as_list(self):
        return np.array(list(self.env.decode(self.state)), dtype=np.float32)

    def get_neighbors(self, heuristic):
        neighbours = []
        allowd_actions = self.info['action_mask']
        for index, action in enumerate(allowd_actions):
            if action:
                neighbours.append(self.action_to_next_state(index))
        neighbours = [TaxiState(state = int(state)) for state in neighbours] 
        cost_by_heuristic = heuristic(neighbours)
        return [(state, cost) for state, cost in zip(neighbours, cost_by_heuristic)]

    
    # def move(self, action):
    #     next_state, reward, terminated, truncated, info = self.env.step(action)
    #     self.state = next_state
    #     return next_state, reward, terminated, truncated, info
    
    def action_to_next_state(self, action):
        """ function to return the neighbor state
            input: action (sent as index of actions array)
            return: next_state
        """
        taxi_row, taxi_col, pass_loc, dest_idx = self.get_state_as_list()
        # - 0: Move south (down)
        # - 1: Move north (up)
        # - 2: Move east (right)
        # - 3: Move west (left)
        # - 4: Pickup passenger
        # - 5: Drop off passenger
        if action == 0:
            taxi_row += 1
        if action == 1:
            taxi_row -= 1
        if action == 2:
            taxi_col += 1
        if action == 3:
            taxi_col -= 1
        if action == 4:
            pass_loc = 4
        if action == 5:
            pass_loc = self.get_passanger_location_by_taxi(taxi_row, taxi_col)

        return self.env.encode(taxi_row, taxi_col, pass_loc, dest_idx)
        # return np.array([taxi_row, taxi_col, pass_loc, dest_idx])

    def get_passanger_location_by_taxi(self, taxi_row, taxi_col):
        # Passenger locations:
        # - 0: Red
        # - 1: Green
        # - 2: Yellow
        # - 3: Blue
        # - 4: In taxi

        if (taxi_row == 0)  &  (taxi_col == 0):
            return 0
        if (taxi_row == 4)  &  (taxi_col == 0):
            return 2
        if (taxi_row == 0)  &  (taxi_col == 4):
            return 1
        if (taxi_row == 4)  &  (taxi_col == 3):
            return 3
        print(f'ilegael drop of at row {taxi_row} and col {taxi_col}')


    # def __len__(self):
    #     return len(self.board)
    def __hash__(self):
        return int(self.state)

    def __eq__(self,state):
        return state == self.state
    
    def __gt__(self,state):
        return False