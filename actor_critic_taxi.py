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
from collections import Counter


SEED = 41


NUM_OF_EPISODES = 2000
LEARNING_RATE = 1e-4
GAMMA = 0.99


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def decide_env_level_and_reset(env, all_rewards, num_of_consecutive_traj):
    """
    reset the taxi env to a random start with changing difficulty levels, based on the sucsses of the model

    input:
        env : taxi_env
        all_rewards : rewards list of all episodes
        num_of_consecutive_steps :  how many consecutive to look back

    returns: state, info (first location)

    difficulty_level == "normal" /  "passanger_and_taxi_in_same_place" / "passanger_in_taxi" / "passanger_and_taxi_in_close_to_testenation"
    """
    options = {}
    if len(all_rewards) <= num_of_consecutive_traj:
        options['difficulty_level'] = "passanger_and_taxi_in_close_to_testenation"
        return env.reset(options = options)
    else: 
        if np.mean(all_rewards[-num_of_consecutive_traj:]) > -50:
            options['difficulty_level'] = "normal"
            return env.reset(options = options)
        elif np.mean(all_rewards[-num_of_consecutive_traj:]) > -70:
            options['difficulty_level'] = "passanger_and_taxi_in_same_place"
            return env.reset(options = options)
        elif np.mean(all_rewards[-num_of_consecutive_traj:]) > -100:
            options['difficulty_level'] = "passanger_and_taxi_in_the_same_location"
            return env.reset(options = options)
        else:
            options['difficulty_level'] = "passanger_and_taxi_in_close_to_testenation"
            return env.reset(options = options)

def get_expected_return(rewards):
    """Compute expected returns per time step."""
    n = len(rewards)
    returns = torch.zeros(n).to(device)
    rewards = rewards.flip(dims=(0,))
    discounted_sum = 0.0
    for i in range(n):
        reward = rewards[i]
        discounted_sum = reward + GAMMA * discounted_sum
        returns[i] = discounted_sum
    returns = returns.flip(dims=(0,))
    return returns.to(device)

def compute_loss(action_probs, values, loss_func, returns):
    """Compute actor-critic loss (combined)."""
    # print(returns.shape, values.shape)
    advantage = returns - values
    # print(advantage.shape)
    action_log_probs = torch.log(action_probs)
    # print(action_log_probs.shape)
    actor_loss = -(action_log_probs * advantage).sum()
    # print(actor_loss)
    critic_loss = loss_func(values, returns)
    # print(critic_loss)
    return actor_loss + critic_loss

def run_episode(env, actorCritic):
    action_probs = []
    values = []
    rewards = []
    actions=[]

    state, info = decide_env_level_and_reset(env, rewards, num_of_consecutive_traj = 5)

    # state, info = env.reset()  # Reset the environment
    state = get_state_as_list(env, state)   # cast to array
    done = False
    while not done:
        
        # get actor net distribution
        distribution, current_state_estimate = actorCritic(torch.from_numpy(state).to(device))
        values.append(current_state_estimate)

        
        # mask to legitimate actions
        # masked_distribution = torch.tensor(info['action_mask']).to(device) * distribution
        masked_distribution = distribution
        action = torch.multinomial(masked_distribution, 1).item()
        actions.append(action)
        action_probs.append(distribution[action])

        # Take the action (a) and observe the outcome state(s') and reward (r)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = get_state_as_list(env, next_state)   # cast int to numpy array
        
        done = terminated or truncated
        rewards.append(reward)

        if done:
            return torch.stack(action_probs, dim=0), torch.cat(values, dim=0), torch.tensor(rewards)
        state = next_state

def run_training(env, actorCritic, device):   
    
    torch.nn.SmoothL1Loss
    actorCritic_optimizer = torch.optim.AdamW(params = actorCritic.parameters(), lr = LEARNING_RATE, weight_decay=1e-4)
    # critic_loss_fn = torch.nn.MSELoss()
    critic_loss_fn = torch.nn.SmoothL1Loss()
    actorCritic_scheduler = StepLR(actorCritic_optimizer, step_size=100, gamma=0.01)
    
    all_rewards = []
    
    for index in range(NUM_OF_EPISODES):
        action_probs, values, rewards = run_episode(env, actorCritic)
        returns = get_expected_return(rewards)#.unsqueeze(1)
        loss = compute_loss(action_probs, values, critic_loss_fn, returns)

        actorCritic_optimizer.zero_grad()
        loss.backward()
        actorCritic_optimizer.step()
        actorCritic_scheduler.step()

        all_rewards.append(sum(rewards))

        if index % 10 == 0:
            print(f'episode - {index}/{NUM_OF_EPISODES}  ---- rewards {np.mean(all_rewards[-10:])}-------')
   
    torch.save(actorCritic.state_dict(),"actorCritic_taxi.pt")


  
       

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
    # env = gym.make('Taxi-v3', render_mode="human")
    
    
    actorCritic = ActorCritic( 
                model_layers = [128, 512, 128],
                ).to(device)
    
    # if os.path.isfile("actorCritic_taxi.pt"):
    #     actorCritic.load_state_dict(torch.load("actorCritic_taxi.pt"))
        
                       
    all_rewards = run_training(env, actorCritic, device)
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

#