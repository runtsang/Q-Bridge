"""Enhanced QCNN model with regularisation and flexible architecture."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNNModel(nn.Module):
    """A convolution‑inspired neural network with dropout, batch‑norm and optional residuals.

    The architecture mirrors the original QCNN but adds modern regularisation
    techniques to improve generalisation on small datasets.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: tuple[int,...] = (16, 16, 12, 8, 4, 4),
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.residual = residual

        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
        )

        layers = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            conv = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Tanh(),
            )
            if self.use_batchnorm:
                conv.append(nn.BatchNorm1d(out_dim))
            conv.append(nn.Dropout(dropout))
            layers.append(conv)
            in_dim = out_dim

        self.convs = nn.ModuleList(layers)

        # Final head
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for conv in self.convs:
            residual = x if self.residual else None
            x = conv(x)
            if self.residual and residual is not None:
                # broadcast if necessary
                if residual.shape!= x.shape:
                    residual = residual[:, : x.shape[1]]
                x = x + residual
        return torch.sigmoid(self.head(x))

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """Simple training loop using binary cross‑entropy loss."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb).squeeze()
                loss = criterion(preds, yb.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss / len(dataset):.4f}")

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions."""
        self.eval()
        with torch.no_grad():
            probs = self(X).squeeze()
        return (probs >= threshold).long()


def QCNN() -> QCNNModel:
    """Factory returning a pre‑configured QCNNModel."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
