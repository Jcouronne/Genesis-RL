import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
from network.ppo import PPO
import os

class PPOAgent:
    def __init__(self, input_dim, output_dim, lr, gamma, clip_epsilon, num_layers, hidden_dim, device, load=False, checkpoint_path=None):
        self.device = device
        
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
        # Set to training mode
        self.model.train()
        
        # Convert all inputs with consistent device placement
        states = torch.stack(states).to(dtype=self.dtype, device=self.device)
        actions = torch.stack(actions).to(dtype=torch.long, device=self.device)
        rewards = torch.stack(rewards).to(dtype=self.dtype, device=self.device)
        dones = torch.stack(dones).to(dtype=torch.bool, device=self.device)
        
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for i, reward in enumerate(reversed(rewards)):
            done_idx = len(rewards) - 1 - i
            R = reward + self.gamma * R * (~dones[done_idx])
            discounted_rewards.insert(0, R)

        discounted_rewards_tensor = torch.stack(discounted_rewards).to(device=self.device)

        # Normalize advantages
        advantages = discounted_rewards_tensor - discounted_rewards_tensor.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        # Update policy using PPO
        for epoch in range(3):
            logits_old = self.model(states).detach()
            probs_old = nn.functional.softmax(logits_old, dim=-1)
            
            logits_new = self.model(states)
            probs_new = nn.functional.softmax(logits_new, dim=-1)

            dist_old = Categorical(probs_old)
            dist_new = Categorical(probs_new)

            ratio = dist_new.log_prob(actions) - dist_old.log_prob(actions)
            ratio = ratio.exp()
            
            # Early stopping check
            if ratio.max() > 2.0 or ratio.min() < 0.5:
                print(f"Early stopping at epoch {epoch}")
                break

            # Calculate surrogate loss
            surrogate_loss_1 = ratio * advantages
            surrogate_loss_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()

            # Optimization with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
    
        # Set back to eval mode for action selection
        self.model.eval()
