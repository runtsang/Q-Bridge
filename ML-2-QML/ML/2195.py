"""
QCNNEnhancedModel: Classical convolution-inspired network with residuals, dropout, and batch norm.

This module extends the original QCNN architecture with modern deep learning practices:
* Residual connections between convolution/pooling layers for better gradient flow.
* Dropout for regularisation.
* BatchNorm to stabilise training.
* A convenience training routine that accepts data loaders.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple

class ResidualBlock(nn.Module):
    """Simple residual block: Linear -> BN -> ReLU -> Dropout -> Linear."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        # If dimensions differ, project the residual
        if in_features!= out_features:
            self.proj = nn.Linear(in_features, out_features)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.ReLU()(self.fc(x) + self.proj(x))

class QCNNEnhancedModel(nn.Module):
    """
    QCNN-like neural network with residual connections.
    Architecture mirrors the original but with added regularisation.
    """
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = ResidualBlock(8, 16, dropout)
        self.conv1 = ResidualBlock(16, 16, dropout)
        self.pool1 = ResidualBlock(16, 12, dropout)
        self.conv2 = ResidualBlock(12, 8, dropout)
        self.pool2 = ResidualBlock(8, 4, dropout)
        self.conv3 = ResidualBlock(4, 4, dropout)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 20,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> Tuple[list[float], list[float]]:
        """
        Train the model using binary cross entropy.
        Returns training and validation losses per epoch.
        """
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device).float()
                optimizer.zero_grad()
                preds = self(x).squeeze()
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)

            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device).float()
                        preds = self(x).squeeze()
                        loss = criterion(preds, y)
                        val_loss += loss.item() * x.size(0)
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f}")

        return train_losses, val_losses

def create_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader | None]:
    dataset = TensorDataset(X, y)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

__all__ = ["QCNNEnhancedModel", "create_dataloaders"]
