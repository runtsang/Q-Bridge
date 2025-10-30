"""Enhanced classical QCNN hybrid model with residual connections and dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class QCNNHybrid(nn.Module):
    """Classical convolution-inspired network with residual connections and dropout.

    The architecture mirrors the original QCNN but adds:
    * BatchNorm layers after each linear transformation.
    * Dropout for regularisation.
    * Residual connections between convolution blocks.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.conv_blocks = nn.ModuleList()
        in_dim = hidden_dim
        for _ in range(3):
            conv = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.BatchNorm1d(in_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.conv_blocks.append(conv)
            # residual connection keeps dimensionality unchanged
            in_dim = in_dim

        self.pool_blocks = nn.ModuleList()
        pool_out_dims = [12, 8, 4]
        in_dim = hidden_dim
        for out_dim in pool_out_dims:
            pool = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.pool_blocks.append(pool)
            in_dim = out_dim

        self.head = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for conv, pool in zip(self.conv_blocks, self.pool_blocks):
            residual = x
            x = conv(x)
            x = pool(x)
            x = x + residual  # residual add
        return torch.sigmoid(self.head(x))

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        """Simple training loop using Adam optimizer and binary cross‑entropy loss."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
            # validation
            self.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                total = 0
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device).float().unsqueeze(1)
                    preds = self(xb)
                    val_loss += criterion(preds, yb).item()
                    preds_cls = (preds > 0.5).float()
                    correct += (preds_cls == yb).sum().item()
                    total += yb.size(0)
                val_loss /= len(val_loader)
                acc = correct / total
                print(f"Epoch {epoch+1}/{epochs} | Val loss: {val_loss:.4f} | Acc: {acc:.4f}")

    @staticmethod
    def synthetic_dataset(
        n_samples: int = 1000,
        input_dim: int = 8,
        noise: float = 0.1,
    ) -> TensorDataset:
        """Generate a toy binary classification dataset."""
        X = torch.randn(n_samples, input_dim)
        # Simple rule: label 1 if sum > 0
        y = (X.sum(dim=1) > 0).float()
        # Add noise
        y = y + noise * torch.randn_like(y)
        y = y.clamp(0, 1)
        return TensorDataset(X, y)


__all__ = ["QCNNHybrid"]
