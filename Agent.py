

import torch
import torch.nn as nn
import torch.nn.functional as F



class Agent_actor(nn.Module):
    def __init__(self, model_layers = [128, 256, 256, 128], input_size = 4):
        super().__init__()
        
        self.actor = nn.Sequential()
        self.actor.add_module('hidden1', nn.Linear(input_size, model_layers[0]))
        self.actor.add_module('relu1', nn.ReLU())
        for i in range(len(model_layers) - 1):
            self.actor.add_module('hidden'+str(i+2), nn.Linear(model_layers[i], model_layers[i+1]))
            self.actor.add_module('relu'+str(i+2), nn.ReLU())
        self.actor.add_module('output', nn.Linear(model_layers[-1], 6))
        
        
    def forward(self, x):
        return F.softmax(self.actor(x), dim = 0)  

    
    

class State_val_net(nn.Module):
    def __init__(self, model_layers = [128, 256, 256, 128], input_size = 4):
        super().__init__()

        self.critic = nn.Sequential()
        self.critic.add_module('hidden1', nn.Linear(input_size, model_layers[0]))
        self.critic.add_module('relu1', nn.ReLU())
        for i in range(len(model_layers) - 1):
            self.critic.add_module('hidden'+str(i+2), nn.Linear(model_layers[i], model_layers[i+1]))
            self.critic.add_module('relu'+str(i+2), nn.ReLU())
        self.critic.add_module('output', nn.Linear(model_layers[-1], 1))
        
        
    def forward(self, x):
        return self.critic(x)  
    

class ActorCritic(nn.Module):
    """Combined actor-critic network."""
    def __init__(self, model_layers=[512], input_size = 4):
        """Initialize."""
        super().__init__()
        self.common = nn.Sequential(
             
        )
        self.common.add_module('hidden1', nn.Linear(input_size, model_layers[0]))
        self.common.add_module('relu1', nn.ReLU())
        for i in range(len(model_layers) - 1):
            self.common.add_module('hidden'+str(i+2), nn.Linear(model_layers[i], model_layers[i+1]))
            self.common.add_module('relu'+str(i+2), nn.ReLU())
        self.actor = nn.Linear(model_layers[-1], 6)
        self.critic = nn.Linear(model_layers[-1], 1)

    def forward(self, inputs):
        x = self.common(inputs)
        return F.softmax(self.actor(x), dim=0), self.critic(x)