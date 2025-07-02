import torch
import random

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, device):
        self.device = device
        self.max_size = max_size
        self.buffer = torch.zeros((max_size, state_dim * 2 + action_dim + 2)).to(self.device)  # State, action, reward, next_state, done
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

    def add(self, state, action, reward, next_state, done):
        transition = torch.cat((state, action.view(-1, 1), reward.view(-1, 1), next_state, done.view(-1, 1)), dim=1)
        
        self.target_ptr = self.ptr + state.shape[0]
        assert self.target_ptr <= 2 * self.max_size
        if self.target_ptr <= self.max_size:
            self.buffer[self.ptr:self.target_ptr] = transition
        else:
            self.buffer[self.ptr:] = transition[:self.max_size - self.ptr]
            self.buffer[:self.target_ptr - self.max_size] = transition[self.max_size - self.ptr:]

        self.ptr = self.target_ptr % self.max_size
        self.size = min(self.size + state.shape[0], self.max_size)

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        batch = self.buffer[indices]
        
        states = batch[:, :self.state_dim]
        actions = batch[:, self.state_dim:self.state_dim + self.action_dim]
        rewards = batch[:, self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1]
        next_states = batch[:, self.state_dim + self.action_dim + 1: -1]
        dones = batch[:, -1]
        
        return states, actions, rewards, next_states, dones

    def size(self):
        return self.size
