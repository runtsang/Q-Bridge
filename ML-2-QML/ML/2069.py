# EstimatorQNN__gen279.py – Classical ML extension

"""
A richer regression model that extends the original 2‑to‑1 feed‑forward network.
Features:
* 3 hidden layers with batch‑norm and dropout.
* Configurable activation (ReLU or Tanh).
* Provides a `train` helper that accepts a DataLoader, optimizer and loss function.
* Returns a fully‑initialized `nn.Module` ready for use in any PyTorch training script.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class EstimatorQNN(nn.Module):
    """
    A flexible regression neural network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dims : Iterable[int], default (64, 32, 16)
        Sizes of the hidden layers.
    activation : str, default "relu"
        Activation function; supports "relu" or "tanh".
    dropout_rate : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (64, 32, 16),
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        act = nn.ReLU() if activation.lower() == "relu" else nn.Tanh()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    act,
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

    # ------------------------------------------------------------------
    # Convenience training helper
    # ------------------------------------------------------------------
    def train_model(
        self,
        loader: DataLoader,
        loss_fn: nn.Module = nn.MSELoss(),
        optimizer: optim.Optimizer | None = None,
        epochs: int = 20,
        device: torch.device | str = "cpu",
    ) -> list[float]:
        """
        Trains the network on the supplied DataLoader.

        Returns
        -------
        losses : list[float]
            Training loss per epoch.
        """
        self.to(device)
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self(xb).squeeze()
                loss = loss_fn(pred, yb.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            losses.append(epoch_loss / len(loader.dataset))
        return losses


__all__ = ["EstimatorQNN"]
