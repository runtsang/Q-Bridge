import numpy as np
import torch
from torch import nn
from typing import Iterable, List

class FCL(nn.Module):
    """
    Extended fully‑connected classical layer.

    This class generalises the original toy example to a multi‑layer
    perceptron that can be trained end‑to‑end with PyTorch.  It
    accepts a vector of parameters ``thetas`` and returns the
    network output as a NumPy array.  The network is automatically
    constructed from ``n_features`` input dimensions, a list of
    hidden layer sizes, and an ``output_dim``.  The ``run`` method
    evaluates the network, and a ``train`` helper performs supervised
    learning with MSE loss.
    """

    def __init__(self,
                 n_features: int = 1,
                 hidden_layers: Iterable[int] | None = None,
                 output_dim: int = 1,
                 device: str | None = None) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")
        layers: List[nn.Module] = []
        inp = n_features
        for h in hidden_layers or []:
            layers.append(nn.Linear(inp, h))
            layers.append(nn.Tanh())
            inp = h
        layers.append(nn.Linear(inp, output_dim))
        self.net = nn.Sequential(*layers).to(self.device)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        x = torch.tensor(list(thetas), dtype=torch.float32, device=self.device).view(-1, 1)
        return self.net(x).mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        return self.forward(thetas).detach().cpu().numpy()

    def train(self,
              data: Iterable[Iterable[float]],
              targets: Iterable[float],
              epochs: int = 100,
              lr: float = 1e-3,
              verbose: bool = False) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        data = list(data)
        targets = torch.tensor(list(targets), dtype=torch.float32, device=self.device)
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = torch.stack([self.forward(d) for d in data]).squeeze(-1)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} loss={loss.item():.4f}")

    def __repr__(self) -> str:
        hidden = [layer.out_features for layer in self.net if isinstance(layer, nn.Linear)][:-1]
        return f"FCL(n_features={self.net[0].in_features}, hidden_layers={hidden}, output_dim={self.net[-1].out_features})"

__all__ = ["FCL"]
