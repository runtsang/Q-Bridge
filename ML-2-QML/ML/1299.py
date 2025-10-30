import numpy as np
import torch
from torch import nn
from typing import Iterable

class FCLayer(nn.Module):
    """
    Extended fully‑connected layer with a hidden ReLU unit.
    Mirrors the original ``FCL`` interface by exposing a ``run`` method that
    accepts a list of parameters and returns a scalar expectation value.
    The network can also be trained with standard PyTorch optimisers.
    """

    def __init__(self, n_features: int = 1, n_hidden: int = 16, n_output: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
        # initialise weights for reproducibility
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimics the seed ``run`` method.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters that will be loaded into the first linear layer
            (flattened). The remaining layers keep their initial weights.

        Returns
        -------
        numpy.ndarray
            Mean of ``tanh`` applied to the network output.
        """
        # Convert to tensor
        params = torch.tensor(list(thetas), dtype=torch.float32)
        # Ensure correct size
        first_layer = self.net[0]
        if params.numel()!= first_layer.weight.numel() + first_layer.bias.numel():
            raise ValueError(
                f"Expected {first_layer.weight.numel() + first_layer.bias.numel()} "
                f"parameters for the first layer, got {params.numel()}"
            )
        # Load parameters
        with torch.no_grad():
            idx = 0
            weight_num = first_layer.weight.numel()
            first_layer.weight.copy_(params[idx:idx+weight_num].view_as(first_layer.weight))
            idx += weight_num
            first_layer.bias.copy_(params[idx:idx+first_layer.bias.numel()].view_as(first_layer.bias))
        # Dummy input
        dummy = torch.randn(1, first_layer.in_features)
        out = self.forward(dummy)
        expectation = torch.tanh(out).mean()
        return expectation.detach().numpy()

    def train_on_synthetic(self, epochs: int = 200, lr: float = 1e-3) -> None:
        """
        Simple training loop on synthetic data to illustrate the class’s
        compatibility with PyTorch optimisers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            x = torch.randn(32, self.net[0].in_features)
            y_true = torch.randn(32, self.net[-1].out_features)
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

__all__ = ["FCLayer"]
