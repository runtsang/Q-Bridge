"""Extended QCNNModel with residual connections and training utilities."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence

class QCNNModel(nn.Module):
    """
    Classical QCNN‑inspired architecture.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dims : Sequence[int]
        Sizes of the successive fully‑connected blocks.
    dropout : float
        Drop‑out probability applied after pooling layers.
    residual : bool
        If True, add skip connections between consecutive conv layers.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Sequence[int] = (16, 16, 12, 8, 4, 4),
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.dropout = nn.Dropout(dropout)

        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
        )

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) // 2):
            in_dim = hidden_dims[2 * i]
            out_dim = hidden_dims[2 * i]
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.Tanh(),
                )
            )

        # Pooling blocks
        self.pool_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) // 2 - 1):
            in_dim = hidden_dims[2 * i]
            out_dim = hidden_dims[2 * i + 2]
            self.pool_blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.Tanh(),
                )
            )

        # Final projection
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.feature_map(x)
        for conv, pool in zip(self.conv_blocks, self.pool_blocks):
            residual = x
            x = conv(x)
            if self.residual:
                x = x + residual
            x = pool(x)
            x = self.dropout(x)
        # Handle the last conv block without following pool
        x = self.conv_blocks[-1](x)
        if self.residual:
            x = x + self.feature_map(x)
        return torch.sigmoid(self.head(x))

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        optimizer: str = "adam",
        loss: str = "mse",
        verbose: bool = True,
    ) -> None:
        """
        Train the model using a simple PyTorch loop.

        Parameters
        ----------
        X : torch.Tensor
            Training inputs of shape (n_samples, input_dim).
        y : torch.Tensor
            Target labels of shape (n_samples, 1).
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        optimizer : str
            Optimizer to use ('adam' or'sgd').
        loss : str
            Loss function ('mse' or 'bce').
        verbose : bool
            Print progress.
        """
        if optimizer == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            opt = torch.optim.SGD(self.parameters(), lr=lr)

        loss_fn = (
            nn.BCELoss() if loss == "bce" else nn.MSELoss()
        )

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.forward(xb)
                loss_val = loss_fn(pred, yb)
                loss_val.backward()
                opt.step()
                epoch_loss += loss_val.item() * xb.size(0)

            epoch_loss /= len(dataset)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs} – Loss: {epoch_loss:.6f}")

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Return predictions for the given inputs."""
        self.eval()
        with torch.no_grad():
            return self.forward(X)

def QCNN() -> QCNNModel:
    """Factory returning a default-configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
