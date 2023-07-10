import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import namedtuple
import time

hidden_size = 128
batch_size = 32

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.002

class Brain():
    def __init__(self, num_hill_points):
        self.agent = BrainAgent(num_hill_points)
        self.num_episode = 0
        self.action = 0
        self.episode_start_time = 0
        self.episode_active = False

    def update_reward(self, score):
        elapsed_time = time.time() - self.episode_start_time
        self.reward = score * 0.1 - elapsed_time * 0.01

    def start_episode(self):
        self.episode_active = True
        self.episode_reward = 0
        self.prev_hill_points = None
        self.prev_bird_position = None
        self.episode_start_time = time.time()
        self.num_episode += 1
        print(f"Starting episode {self.num_episode}")

    def end_episode(self):
        if self.episode_active == False:
            return
        self.episode_active = False
        print(f"Episode: {self.num_episode}, Total Reward: {self.episode_reward}")
        self.agent.update_epsilon(self.num_episode)
        self.episode_start_time = time.time()

    def add_observation(self, hill_points, bird_position):
        if self.prev_hill_points == None or self.prev_bird_position == None:
            self.prev_hill_points = hill_points
            self.prev_bird_position = bird_position
            return
        
        self.prev_action = self.action
        self.action = self.agent.select_action(hill_points, bird_position)

        self.agent.remember(self.prev_hill_points, self.prev_bird_position, self.prev_action, hill_points, bird_position, self.reward)
        self.agent.learn()
        self.episode_reward += self.reward

        self.prev_hill_points = hill_points
        self.prev_bird_position = bird_position

    def save(self):
        torch.save(self.model.state_dict(), 'models/brain.pth')

# Define the deep Q-network model
class BrainDQN(nn.Module):
    def __init__(self, num_hill_points, hidden_size, output_size):
        super(BrainDQN, self).__init__()

        self.hill_encoder = nn.Sequential(
            nn.Linear(num_hill_points, hidden_size),
            nn.ReLU()
        )
        self.bird_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, hill_points, bird_location):
        hill_encoded = self.hill_encoder(hill_points)
        bird_encoded = self.bird_encoder(bird_location)
        combined = torch.cat((hill_encoded, bird_encoded), dim=1)
        output = self.fc(combined)
        return output
    

class BrainAgent:
    def __init__(self, num_hill_points):
        self.model = BrainDQN(num_hill_points, hidden_size, 2) # Output size of 2 to represent "press" or "not press"
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) # Learning rate = 0.001
        self.memory = ReplayMemory(10000)
        self.epsilon = epsilon_start

    def select_action(self, hill_points, bird_location):
        if random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(hill_points), torch.FloatTensor(bird_location))
                action = q_values.argmax().item()
        return action

    def remember(self, hill_points, bird_location, action, next_hill_points, next_bird_location, reward):
        transition = (hill_points, bird_location, action, next_hill_points, next_bird_location, reward)
        self.memory.push(transition)

    def learn(self):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))

        hill_points_batch = torch.FloatTensor(batch[0])
        bird_location_batch = torch.FloatTensor(batch[1])
        action_batch = torch.LongTensor(batch[2])
        next_hill_points_batch = torch.FloatTensor(batch[3])
        next_bird_location_batch = torch.FloatTensor(batch[4])
        reward_batch = torch.FloatTensor(batch[5])

        current_q_values = self.model(hill_points_batch, bird_location_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.model(next_hill_points_batch, next_bird_location_batch).max(1)[0].detach()
        target_q_values = reward_batch + 0.99 * next_q_values # Gamma 0.99

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, num_episode):
        self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                       math.exp(-1.0 * num_episode / epsilon_decay)


# Define the replay memory class
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)