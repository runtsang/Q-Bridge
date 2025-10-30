import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """
    A hybrid multi-layer perceptron that mimics a fully connected quantum layer.
    Supports configurable depth, dropout, and L2 regularisation.
    """
    def __init__(self, n_features: int = 1, n_hidden: int = 32, n_layers: int = 3,
                 dropout: float = 0.1, l2_reg: float = 1e-4):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, n_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = n_hidden
        layers.append(nn.Linear(n_hidden, 1))
        self.model = nn.Sequential(*layers)
        self.l2_reg = l2_reg

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        x = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.model(x)
        return out.squeeze()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        with torch.no_grad():
            out = self.forward(thetas)
        return out.detach().cpu().numpy()
