# from utilities import *
from heuristics import BellmanUpdateHeuristic, BootstrappingHeuristic
from taxi_wrapper import ExtendedTaxiEnv, DetermenisticTaxiEnv
from BWAS import BWAS
import math
import random
import pickle
import numpy as np
from taxi_state import TaxiState
import os



def bootstrappingTraining(bootstrapping_heuristic):
    
    size=500000
    batch_size = 128 
    W = 5
    B = 10
    train_loss = []
    if os.path.isfile("state_distance_dict.pickle"):
        with open("state_distance_dict.pickle", "rb") as f:
                state_distance_dict = pickle.load(f)
    else:
        state_distance_dict = {}

    for num_of_batches in range(20, math.ceil(size/batch_size)): 
        
        batch = []
        
        while len(batch) < batch_size:
            start_state = TaxiState()

            T = 32
            path_to_goal, _ = BWAS(start_state, W, B, bootstrapping_heuristic.get_h_values, T)
            while not path_to_goal:
                T *= 2
                path_to_goal, _ = BWAS(start_state, W, B, bootstrapping_heuristic.get_h_values, T)
                
            path_to_goal.reverse()
            
            for distance, state in enumerate(path_to_goal):
                batch.append((state, distance))
                
                state_distance_dict = update_state_distance_dict(distance, state, state_distance_dict)
        
        
        for state, distance in state_distance_dict.items():
            state = [int(num) for num in state.strip('[]').split(',')]
            state = TaxiState(state = int(encode(state))) 
            batch.append((state, distance))

        input_data = [state for state, _ in batch]
        output_labels = [distance for _, distance in batch]
        print(f'batch {num_of_batches} / {math.ceil(size/batch_size)} current distanse {num_of_batches+2}')
        training_loss = bootstrapping_heuristic.train_model(input_data, output_labels, epochs=5)
        train_loss.append(training_loss)
        
        bootstrapping_heuristic.save_model()
        with open("state_distance_dict.pickle", "wb") as f:
            pickle.dump(state_distance_dict, f)
        with open("train_loss.pickle", "wb") as f:
            pickle.dump(train_loss, f)
    
    


def update_state_distance_dict(distance, state, state_dict) :
    state = list(state.get_state_as_list().astype(int))
    if (repr(state) not in state_dict) or (distance < state_dict[repr(state)]):
        state_dict[repr(state)] = distance
        print(f'updated/inserted {state}. and now is {distance}')
        
    
    return state_dict

def encode(state):
    taxi_row, taxi_col, pass_loc, dest_idx = state
    i = taxi_row
    i *= 5
    i += taxi_col
    i *= 5
    i += pass_loc
    i *= 4
    i += dest_idx
    return i