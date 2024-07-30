from heuristics import BellmanUpdateHeuristic, BootstrappingHeuristic, BaseHeuristic
from BWAS import BWAS
from taxi_state import TaxiState
from training import bootstrappingTraining
import time
import numpy as np
import pickle
from taxi_wrapper import ExtendedTaxiEnv, DetermenisticTaxiEnv
from gymnasium.wrappers import TimeLimit
import torch
import pandas as pd

def train_bootstrapping():
    now = time.time()
    BS_heuristic = BootstrappingHeuristic(4, 4)
    BS_heuristic.load_model()
    bootstrappingTraining(BS_heuristic)
    print(f'train time bootstrapping = {time.time() - now}')

def test_bootstrapping():
    path_length = []
    W_s = []
    B_s = []
    expansions_s = []
    run_time = []
    heuristics = []

    BS_heuristic = BootstrappingHeuristic(4, 4)
    BS_heuristic.load_model()
    base_heuristic = BaseHeuristic()

    T = 1000
    
    with torch.no_grad():
        for W in [2,5]:
            for B in [1,100]:
                for heuristic in [base_heuristic, BS_heuristic]:
                    for _ in range(10):
                        start_state = TaxiState()
                        start_time = time.time()
                        path_to_goal, expansions = BWAS(start_state, W, B, heuristic.get_h_values, T)
                        run_time.append(time.time()-start_time)
                        if path_to_goal:
                            heuristics.append(str(type(heuristic)))
                            path_length.append(len(path_to_goal))
                            W_s.append(W)
                            B_s.append(B)
                            expansions_s.append(expansions)
                        else:
                            heuristics.append(str(type(heuristic)))
                            path_length.append(200)
                            W_s.append(W)
                            B_s.append(B)
                            expansions_s.append(expansions)
        
    results = {"w" : W_s,
                "B" : B_s,
                "Runtime" : run_time,
                "Heuristic" : heuristics,
                "Path_length" : path_length,
               "Expansions" : expansions_s,

               }
    
    df = pd.DataFrame(results)
    
    with open("bwas_results2.pickle", "wb") as f:
        pickle.dump(df, f)

  
   

def main():
    np.random.seed(42)
    # train_bootstrapping()
    test_bootstrapping()
   


if __name__ == "__main__":
    main()

