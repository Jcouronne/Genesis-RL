import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(PPO, self).__init__()

        # Define the sequence of layers
        layers = []

        # Initial dense layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(Swish())

        # Add N repeated blocks of Dense, LayerNorm, and Swish
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(Swish())

        # Final dense layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Create the sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


