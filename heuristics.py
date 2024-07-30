import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math

class BaseHeuristic:
    def init(self):
        pass

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        gaps = []
        location_dict = {0:(0, 0), 1:(0, 4), 2:(4, 0), 3:(4, 3)}
        for state_as_list in states_as_list:
            taxi_row, taxi_col, passenger_loc, goal_loc = state_as_list
            goal_row, goal_col = location_dict[goal_loc]
            
            if passenger_loc == 4:
                passenger_row, passenger_col = taxi_row, taxi_col
            else:
                passenger_row, passenger_col = location_dict[passenger_loc]

            distance_taxi_to_passenger = math.sqrt((passenger_row - taxi_row)**2 + (passenger_col - taxi_col)**2)
            distance_passenger_to_goal = math.sqrt((goal_row - passenger_row)**2 + (goal_col - passenger_col)**2)
            
            gap = distance_taxi_to_passenger + distance_passenger_to_goal + 1
            gaps.append(gap)
        
        return gaps

class HeuristicModel(nn.Module):
    def __init__(self, input_dim):
        super(HeuristicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 256)
        # self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class LearnedHeuristic:
    def __init__(self, n=4, k=4):
        self._n = n
        self._k = k
        self._model = HeuristicModel(n)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.005, weight_decay = 1e-02)

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        states = np.array(states_as_list, dtype=np.float32)
        states_tensor = torch.tensor(states)
        with torch.no_grad():
            predictions = self._model(states_tensor).numpy()
        return -predictions.flatten()

    def train_model(self, input_data, output_labels, epochs=5):
        input_as_list = [state.get_state_as_list() for state in input_data]
        inputs = np.array(input_as_list, dtype=np.float32)
        outputs = np.array(output_labels, dtype=np.float32)

        # Shuffle the data
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        inputs = inputs[indices]
        outputs = outputs[indices]
        batch_size = 1024
        # num_of_batches = math.floor(len(inputs/batch_size))
        inputs_tensor = torch.tensor(inputs)
        outputs_tensor = torch.tensor(outputs).unsqueeze(1)  # Adding a dimension for the output
        training_loss = 0
        self._model.train()
        for _ in range(epochs):
            for start_idx in range(0, len(inputs), batch_size):
                
                self._optimizer.zero_grad()

                predictions = self._model(inputs_tensor[start_idx:start_idx+batch_size])
                loss = self._criterion(predictions, outputs_tensor[start_idx:start_idx+batch_size])
                loss.backward()
                self._optimizer.step()
                training_loss += loss.item()
        training_loss /= len(inputs)
        print(f'------------------------- loss {training_loss} ----------------------')
        return training_loss

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self._model.load_state_dict(torch.load(path))
            self._model.eval()

class BellmanUpdateHeuristic(LearnedHeuristic):
    def __init__(self, n=4, k=4):
        super().__init__(n, k)
        
    def save_model(self):
        super().save_model('bellman_update_heuristic.pth')

    def load_model(self):
        super().load_model('bellman_update_heuristic.pth')

class BootstrappingHeuristic(LearnedHeuristic):
    def __init__(self, n=4, k=4):
        super().__init__(n, k)
        
    def save_model(self):
        super().save_model('bootstrapping_heuristic.pth')

    def load_model(self):
        super().load_model('bootstrapping_heuristic.pth')
