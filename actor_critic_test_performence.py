import numpy as np
from tqdm import tqdm
import os
import gymnasium as gym
import torch
from Agent import Agent_actor, State_val_net, ActorCritic
import pickle
from torch.optim.lr_scheduler import StepLR
import random
import time
import sys
sys.path.append(r"C:\Users\yaron\projects\BGU - DecisionMaking\project\bwas bootstraping")
from taxi_wrapper import ExtendedTaxiEnv
from gymnasium.wrappers import TimeLimit






SEED = 41
NUM_OF_EPISODES = 10 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 



def test_model(env, actorCritic, device):
    
    all_rewards = []
    
    options = {}
    # options['difficulty_level'] = "normal"
    # options['difficulty_level'] = "passanger_in_taxi"
    # options['difficulty_level'] = "passanger_and_taxi_in_same_place"
    options['difficulty_level'] = "taxi close to destenation"
    actorCritic.eval()

    for episode in range(NUM_OF_EPISODES):
        
        state, info = env.reset(options = options)  # Reset the environment
        state = get_state_as_list(env, state)   # cast to array
        episode_rewards = 0
        done = False
        while not done:
            with torch.no_grad():
                # get actor net distribution
                distribution, _ = actorCritic(torch.from_numpy(state).to(device))
                masked_distribution = torch.tensor(info['action_mask']).to(device) * distribution
                action = torch.multinomial(masked_distribution, 1).item()
            
                # Take the action (a) and observe the outcome state(s') and reward (r)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = get_state_as_list(env, next_state)   # cast int to numpy array
                # env.render()
                done = terminated or truncated
                episode_rewards += reward
             
                if done:
                    print(episode_rewards)
                    all_rewards.append(episode_rewards)
                    break
                state = next_state

        if episode % 10 == 0:
            print(f'episode - {episode}/{NUM_OF_EPISODES}  ---- rewards {np.mean(all_rewards[-10:])}-------')

    return all_rewards

def main():

    start = time.process_time()

    # set seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # env = TimeLimit(env = ExtendedTaxiEnv(render_mode="human"), max_episode_steps = 200)
    env = TimeLimit(env = ExtendedTaxiEnv(), max_episode_steps = 200)
    
    
    actorCritic = ActorCritic( 
                model_layers = [128, 512, 128],
                ).to(device)
    
    if os.path.isfile("actorCritic_taxi.pt"):
        actorCritic.load_state_dict(torch.load("actorCritic_taxi.pt"))
        
                      
    all_rewards = test_model(env, actorCritic, device)
    with open('all_rewards.pickle','wb') as f:
        pickle.dump(all_rewards, f)

    print(time.process_time() - start)
    
def is_goal_state(state):
    if state in [0,85,410,495]:
        return True
    else:
        return False
    
def get_state_as_list(env, state):
    return np.array(list(env.decode(state)), dtype=np.float32)

def get_neighbours(env, state, info):
    neighbours = []
    allowd_actions = info['action_mask']
    for index, action in enumerate(allowd_actions):
        if action:
          neighbours.append(action_to_next_state(env, state, index))  
    return neighbours

def action_to_next_state(env, state, action):
    taxi_row, taxi_col, pass_loc, dest_idx = get_state_as_list(env, state)
    # - 0: Move south (down)
    # - 1: Move north (up)
    # - 2: Move east (right)
    # - 3: Move west (left)
    # - 4: Pickup passenger
    # - 5: Drop off passenger
    if action == 0:
        taxi_row -= 1
    if action == 1:
        taxi_row += 1
    if action == 2:
        taxi_col += 1
    if action == 3:
        taxi_col -= 1
    if action == 4:
        pass_loc = 4
    if action == 5:
        pass_loc = get_passanger_location_by_taxi()

    return env.encode(taxi_row, taxi_col, pass_loc, dest_idx)

def get_passanger_location_by_taxi(taxi_row, taxi_col):
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


if __name__ == "__main__":
    main()


