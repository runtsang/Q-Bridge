import numpy as np
import torch
from torch import nn
from torch.nn import Dropout

class FCL(nn.Module):
    """
    Extended fully connected layer with dropout regularization.
    The `run` method accepts a list of parameters (thetas) and returns
    a 1‑D NumPy array of the layer output.
    """
    def __init__(self, n_features: int = 1, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout_prob)
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: list[float]) -> np.ndarray:
        tensor = torch.tensor(thetas, dtype=torch.float32).view(-1, 1)
        out = self.linear(self.dropout(tensor))
        return torch.tanh(out).detach().numpy().flatten()

__all__ = ["FCL"]
