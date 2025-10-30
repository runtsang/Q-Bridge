import numpy as np
import torch
from torch import nn
from typing import Iterable

class HybridFCL(nn.Module):
    """
    A multi‑layer fully‑connected network that mimics the behaviour of a quantum
    fully‑connected layer while allowing classical training and inference.
    Parameters are supplied as a flat iterable via the ``run`` method.
    """
    def __init__(self, n_features: int = 1, hidden_sizes: Iterable[int] = (32, 16),
                 output_size: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def _flatten_params(self):
        """Return a flat list of all learnable parameters."""
        return [p.detach().cpu().numpy().flatten() for p in self.parameters()]

    def set_parameters(self, params: Iterable[float]):
        """Set the network weights from a flat iterable."""
        flat = np.asarray(list(params), dtype=np.float32)
        offset = 0
        for p in self.parameters():
            shape = p.shape
            size = p.numel()
            new = flat[offset:offset + size].reshape(shape)
            p.data.copy_(torch.from_numpy(new))
            offset += size

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass with externally supplied parameters.
        Returns the mean activation of the last layer as a 1‑D array.
        """
        self.set_parameters(thetas)
        with torch.no_grad():
            x = torch.ones(1, self.net[0].in_features)
            out = self.net(x)
            mean_out = out.mean(dim=0)
            return mean_out.detach().cpu().numpy()
