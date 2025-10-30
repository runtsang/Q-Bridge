"""Enhanced fully connected neural network with training utilities."""

import numpy as np
import torch
from torch import nn, optim
from typing import Iterable, List, Tuple

class FullyConnectedLayer(nn.Module):
    """A multi‑layer perceptron with optional dropout and batch‑norm.

    The network is fully parameterised by a flat array of weights and biases.
    The ``run`` method accepts a list/array of parameters, loads them into the
    model, evaluates a single forward pass on a fixed dummy input and returns
    the mean of the output as a NumPy array.  This mirrors the behaviour of
    the original quantum example while offering richer expressivity.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # not after output layer
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(activation)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def set_parameters_from_flat(self, theta: Iterable[float]) -> None:
        """Load a flat list of parameters into the network."""
        flat_params = torch.tensor(list(theta), dtype=torch.float32)
        with torch.no_grad():
            idx = 0
            for param in self.parameters():
                numel = param.numel()
                param.copy_(flat_params[idx: idx + numel].view_as(param))
                idx += numel

    def run(self, theta: Iterable[float]) -> np.ndarray:
        """Evaluate the network on a fixed dummy input and return the mean."""
        # Load parameters
        self.set_parameters_from_flat(theta)

        # Dummy input: ones of shape (1, input_dim)
        dummy = torch.ones(1, self.model[0].in_features)
        out = self(dummy)
        # Return mean of output as numpy array
        return out.mean().detach().numpy()

    def train_on_synthetic(
        self,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 42
    ) -> Tuple[np.ndarray, List[float]]:
        """Simple training loop on synthetic data.

        Returns the final parameters and a list of training losses.
        """
        torch.manual_seed(seed)
        # Synthetic regression data: y = sin(x)
        x = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
        y = torch.sin(x)
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []

        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())

        # Return flattened parameters
        flat_params = torch.cat([p.view(-1) for p in self.parameters()]).detach().numpy()
        return flat_params, losses

def FCL() -> FullyConnectedLayer:
    """Convenience factory matching the original API."""
    return FullyConnectedLayer()

__all__ = ["FullyConnectedLayer", "FCL"]
