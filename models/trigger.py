# models/trigger.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TriggerGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        A simple MLP that learns a trigger pattern.
        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Dimensionality of the hidden layer.
        """
        super(TriggerGenerator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.mlp(x)
