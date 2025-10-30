import numpy as np
import torch
from torch import nn
from torch.optim import Adam

class FullyConnectedLayer(nn.Module):
    """
    An extended fully connected layer with two hidden units and ReLU activation.
    Supports batched input and a lightweight training helper.
    """
    def __init__(self, n_features: int = 1, hidden: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batched tensors.
        """
        return self.net(thetas)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the original run signature: accepts an iterable of scalars,
        converts to a batched tensor, and returns the mean activation.
        """
        batch = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.forward(batch)
        return out.mean(dim=0).numpy()

    def train_step(self, thetas: Iterable[float], targets: Iterable[float],
                   lr: float = 1e-3) -> float:
        """
        A single gradient update step using MSE loss.
        """
        batch = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        target = torch.as_tensor(list(targets), dtype=torch.float32).view(-1, 1)
        self.train()
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        optimizer.zero_grad()
        pred = self.forward(batch)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()

def FCL() -> FullyConnectedLayer:
    """
    Factory returning an instance of the extended fully connected layer.
    """
    return FullyConnectedLayer()

__all__ = ["FCL", "FullyConnectedLayer"]
