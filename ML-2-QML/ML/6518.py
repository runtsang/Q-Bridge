import torch
import torch.nn as nn
from typing import Iterable

class FCL(nn.Module):
    """
    Multi‑layer fully connected network that replaces the single linear layer
    from the seed.  It exposes a ``run`` method that accepts an iterable of
    input values and returns a single output value as a NumPy array.
    """
    def __init__(self, n_features: int = 1, hidden_sizes=(32, 16)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Convert the input iterable to a tensor, forward it through the network
        and return the scalar output as a NumPy array.
        """
        x = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.net(x)
        return out.detach().numpy()

__all__ = ["FCL"]
