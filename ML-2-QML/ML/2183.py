import torch
from torch import nn
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """
    Classical fully‑connected layer with two hidden layers, dropout, and a training helper.
    The ``run`` method treats the input iterable as a collection of samples.
    """
    def __init__(self, n_features: int = 1, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accepts an iterable of floats, reshapes them into batches of ``n_features``,
        runs the network, and returns the mean activation as a NumPy array.
        """
        data = torch.as_tensor(list(thetas), dtype=torch.float32)
        if self.net[0].in_features > 1:
            if data.numel() % self.net[0].in_features!= 0:
                raise ValueError("Input length must be a multiple of n_features")
            data = data.view(-1, self.net[0].in_features)
        else:
            data = data.view(-1, 1)

        out = self.forward(data)
        expectation = out.mean().item()
        return np.array([expectation])

    def train_step(self, thetas: Iterable[float], target: float, lr: float = 1e-3) -> float:
        """
        Performs one Adam optimization step on the network, minimizing the squared error
        between the network output and a scalar target.
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        data = torch.as_tensor(list(thetas), dtype=torch.float32)
        if self.net[0].in_features > 1:
            data = data.view(-1, self.net[0].in_features)
        else:
            data = data.view(-1, 1)

        optimizer.zero_grad()
        out = self.forward(data)
        loss = (out - target) ** 2
        loss.mean().backward()
        optimizer.step()
        return loss.item()

__all__ = ["FCL"]
