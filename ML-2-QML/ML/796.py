"""
Hybrid classical convolutional filter with data‑augmentation and early‑stopping.
Replaces the original ConvFilter but keeps the same ``Conv()`` factory.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class ConvFilter(nn.Module):
    """Convolutional filter with thresholded sigmoid activation."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 augment: bool = False, early_stop_patience: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.augment = augment
        self.early_stop_patience = early_stop_patience
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """Forward pass on a single 2‑D array."""
        if isinstance(data, np.ndarray):
            data = torch.as_tensor(data, dtype=torch.float32)
        data = data.view(1, 1, self.kernel_size, self.kernel_size)
        with torch.no_grad():
            return self.forward(data).item()

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random shift, flip and scaling augmentation."""
        # Random horizontal flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        # Random vertical flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-2])
        # Random shift
        shift = random.randint(-1, 1)
        if shift!= 0:
            x = torch.roll(x, shifts=shift, dims=-1)
        return x

    def train_on(self, dataset: Iterable[Tuple[np.ndarray, float]],
                 epochs: int = 50, lr: float = 0.01) -> None:
        """Simple training loop with optional data augmentation and early stopping.

        Args:
            dataset: Iterable of (image, target) pairs. Images are 2‑D arrays.
            epochs: Maximum number of epochs.
            lr: Learning rate.
        """
        # Convert dataset to tensors
        imgs, targets = zip(*dataset)
        imgs = torch.stack([torch.as_tensor(img, dtype=torch.float32)
                            for img in imgs]).unsqueeze(1)
        targets = torch.as_tensor(targets, dtype=torch.float32).unsqueeze(1)

        if self.augment:
            imgs = torch.stack([self._augment(img) for img in imgs])

        data_loader = DataLoader(TensorDataset(imgs, targets), batch_size=16, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = math.inf
        patience = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_imgs, batch_targets in data_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_imgs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_imgs.size(0)

            epoch_loss /= len(data_loader.dataset)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def evaluate(self, dataset: Iterable[Tuple[np.ndarray, float]]) -> float:
        """Compute mean squared error on a dataset."""
        imgs, targets = zip(*dataset)
        imgs = torch.stack([torch.as_tensor(img, dtype=torch.float32)
                            for img in imgs]).unsqueeze(1)
        targets = torch.as_tensor(targets, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            outputs = self.forward(imgs)
            mse = nn.functional.mse_loss(outputs, targets)
        return mse.item()

    def set_threshold(self, threshold: float) -> None:
        """Update the threshold used in the sigmoid activation."""
        self.threshold = threshold

    def get_params(self) -> dict:
        """Return the current parameters of the filter."""
        return {k: v.detach().cpu().numpy() for k, v in self.named_parameters()}

def Conv(kernel_size: int = 2, threshold: float = 0.0,
         augment: bool = False, early_stop_patience: int = 5) -> ConvFilter:
    """Factory function that returns a ConvFilter instance."""
    return ConvFilter(kernel_size=kernel_size,
                      threshold=threshold,
                      augment=augment,
                      early_stop_patience=early_stop_patience)

__all__ = ["Conv", "ConvFilter"]
