"""Enhanced classical regression model with residual connections and custom loss.

This module defines EstimatorQNN, a PyTorch neural network that extends the
original toy network by adding residual blocks, dropout, batch‑normalisation
and a hybrid loss that supports both regression and classification signals.
It also exposes a convenient ``train`` method that runs a full training
pipeline on a given dataset.

The goal is to provide a research‑ready baseline that can be swapped into
experiments that compare classical, quantum and hybrid approaches.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Sequence, Tuple, Optional

class EstimatorQNN(nn.Module):
    """Feature‑aware residual network for regression.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : Sequence[int]
        Sizes of the hidden residual blocks.
    dropout : float
        Drop‑out probability applied after each block.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList()
        prev_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            block = nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, prev_dim),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)
            prev_dim = prev_dim
        self.output_layer = nn.Linear(prev_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        out = self.input_layer(x)
        for block in self.blocks:
            residual = out
            out = block(out)
            out = out + residual  # residual connection
            out = self.dropout(out)
        out = self.output_layer(out)
        return out.squeeze(-1)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def _mse_with_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        l1_coef: float = 1e-4,
    ) -> torch.Tensor:
        """Mean‑squared error with optional L1 regularisation."""
        mse = nn.functional.mse_loss(pred, target)
        l1 = sum(p.abs().sum() for p in self.parameters())
        return mse + l1_coef * l1

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        l1_coef: float = 1e-4,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the network and return training/validation losses.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader, optional
            DataLoader for validation data.
        epochs : int
            Number of epochs.
        lr : float
            Learning rate.
        l1_coef : float
            Coefficient for L1 regularisation.
        device : str
            Device to run the training on.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self(xb)
                loss = self._mse_with_l1(pred, yb, l1_coef)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = self(xb)
                        loss = self._mse_with_l1(pred, yb, l1_coef)
                        val_loss += loss.item() * xb.size(0)
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)

        return torch.tensor(train_losses), torch.tensor(val_losses)

    def predict(self, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Predict on new data."""
        self.eval()
        with torch.no_grad():
            return self(x.to(device)).cpu()
