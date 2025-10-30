import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List

class FCL(nn.Module):
    """
    Multi‑layer fully connected network that mimics the quantum FCL interface.
    The public `run` method accepts a list of theta values, feeds them as a
    single feature vector, and returns the mean of the final layer output.
    """

    def __init__(self, n_features: int = 1, hidden_sizes: List[int] = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [8, 4]
        layers = [nn.Linear(n_features, hidden_sizes[0]), nn.ReLU()]
        for h_in, h_out in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers += [nn.Linear(h_in, h_out), nn.ReLU()]
        layers += [nn.Linear(hidden_sizes[-1], 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        return self.model(thetas)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the network on the supplied theta vector and return a NumPy array."""
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.forward(theta_tensor).mean()
        return output.item()

    def train_step(self, thetas: Iterable[float], target: float, lr: float = 1e-3):
        """Single gradient‑descent step for demonstration purposes."""
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32, requires_grad=True).unsqueeze(0)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss = (self.forward(theta_tensor) - target) ** 2
        loss.backward()
        optimizer.step()
        return loss.item()
