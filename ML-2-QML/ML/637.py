"""Configurable feed‑forward regressor with dropout, batchnorm and early‑stopping.

Designed to replace the tiny 3‑layer network in the seed example while
remaining fully compatible with the original EstimatorQNN interface.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Sequence, Optional, Tuple

class EstimatorQNN(nn.Module):
    """
    A flexible regression network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dims : Sequence[int]
        Sizes of hidden layers.
    dropout : float | None
        Dropout probability applied after each hidden layer. If None, dropout is disabled.
    batchnorm : bool
        Whether to include BatchNorm1d after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (8, 4),
        dropout: Optional[float] = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Tanh())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    # ------------------------------------------------------------------
    # Training helpers (not part of the original seed but useful in practice)
    # ------------------------------------------------------------------
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 20,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, list[float]]:
        """
        Train the network using MSE loss and Adam optimizer.

        Returns
        -------
        history : list[float]
            Training loss history.
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        history: list[float] = []
        best_loss = float("inf")
        best_state = None
        counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.forward(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1:03d} loss={epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping after {epoch+1} epochs.")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return torch.tensor(history), history

    def predict(self, X: torch.Tensor, *, batch_size: int = 32) -> torch.Tensor:
        """
        Predict outputs for given inputs.
        """
        self.eval()
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, in loader:
                preds.append(self.forward(xb))
        return torch.cat(preds, dim=0)

__all__ = ["EstimatorQNN"]
