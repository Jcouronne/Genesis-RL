import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
from network.ppo import PPO
import os

class PPOAgent:
    def __init__(self, input_dim, output_dim, lr, gamma, clip_epsilon, num_layers, device, load=False, num_envs=1, hidden_dim=64, checkpoint_path=None):
        self.device = device
        self.num_envs = num_envs
        
        # Create model and set precision based on the current gs.init precision
        self.model = PPO(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        
        # Get the current precision from genesis settings
        self.dtype = torch.float64
        
        # Convert model parameters to the correct precision
        self.model = self.model.to(dtype=self.dtype)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.checkpoint_path = checkpoint_path
        
        if load and os.path.exists(checkpoint_path):
            self.load_checkpoint()

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state successfully loaded")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state successfully loaded")
        
        self.model.eval()  # Set to evaluation mode
        print(f"Checkpoint loaded from {self.checkpoint_path}")

    def select_action(self, state):
        # Ensure state has the right precision
        state = state.to(dtype=self.dtype)
        
        with torch.no_grad():
            logits = self.model(state)
        probs = nn.functional.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action

    def train(self, states, actions, rewards, dones):
        # Convert all inputs to the right precision
        states = torch.stack(states).to(dtype=self.dtype)
        actions = torch.stack(actions).to(dtype=self.dtype)
        rewards = torch.stack(rewards).to(dtype=self.dtype)
        dones = torch.stack(dones).to(dtype=torch.bool)
        
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R * (~dones[-1])
            discounted_rewards.insert(0, R)

        discounted_rewards_tensor = torch.stack(discounted_rewards).to(self.device)

        # Normalize the rewards
        advantages = discounted_rewards_tensor - discounted_rewards_tensor.mean()
        
        # Update policy using PPO
        for _ in range(10):  # Number of epochs for each batch update
            logits_old = self.model(states).detach()
            probs_old = nn.functional.softmax(logits_old, dim=-1)
            
            logits_new = self.model(states)
            probs_new = nn.functional.softmax(logits_new, dim=-1)

            dist_old = Categorical(probs_old)
            dist_new = Categorical(probs_new)

            ratio = dist_new.log_prob(actions) - dist_old.log_prob(actions)
            ratio = ratio.exp()

            # Calculate surrogate loss
            surrogate_loss_1 = ratio * advantages
            surrogate_loss_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()

            # Perform optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
