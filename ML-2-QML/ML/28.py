import numpy as np
import torch
from torch import nn, optim
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """
    Classical fully‑connected layer that mimics the behaviour of the original
    quantum example while providing a full PyTorch training loop.
    """
    def __init__(self, n_features: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(n_features, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a single output value."""
        x = self.activation(self.linear(thetas))
        return self.output(x).mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Public API that accepts an iterable of parameters and returns a NumPy
        array, matching the original seed's interface.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            out = self.forward(tensor)
        return out.detach().numpy()

    def train_step(self, thetas: Iterable[float], target: float,
                   optimizer: optim.Optimizer, loss_fn=nn.MSELoss()) -> float:
        """
        One optimisation step: forward, loss, backward and parameter update.
        """
        optimizer.zero_grad()
        preds = self.forward(torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1))
        loss = loss_fn(preds, torch.tensor([target], dtype=torch.float32))
        loss.backward()
        optimizer.step()
        return loss.item()

def FCL() -> FullyConnectedLayer:
    """
    Factory function mirroring the original seed interface.
    """
    return FullyConnectedLayer()
