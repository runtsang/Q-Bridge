"""Enhanced classical convolutional filter with residual connections and a deeper classifier.

This module defines `QuanvolutionNet`, a PyTorch model that replaces the original
simple 2×2 convolution with a residual block and a two‑layer MLP head.  The
network can be trained with the `fit` method for quick experimentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional

__all__ = ["QuanvolutionNet"]


class ResidualBlock(nn.Module):
    """A lightweight residual block operating on the output of the 2×2 conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv(x)
        out = self.bn(residual)
        out = self.relu(out)
        return x + out


class QuanvolutionNet(nn.Module):
    """Classical quanvolution network with residual connections and a deeper classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 2×2 stride‑2 conv that reduces a 28×28 image to 14×14 feature maps
        self.base_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        # Residual block that operates on the 4‑channel feature maps
        self.res_block = ResidualBlock(4, 4)
        # Two‑layer MLP head
        self.classifier = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_conv(x)
        x = self.res_block(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Convenience training helper
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Simple one‑shot training loop.

        Parameters
        ----------
        train_loader : Iterable
            DataLoader yielding (inputs, targets).
        val_loader : Optional[Iterable], default None
            Validation DataLoader.
        epochs : int, default 10
            Number of epochs.
        lr : float, default 1e-3
            Learning rate.
        device : str, default "cpu"
            Device to run on.
        verbose : bool, default True
            Whether to print progress.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.NLLLoss()

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                log_probs = self(xb)
                loss = criterion(log_probs, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            if verbose:
                avg_loss = epoch_loss / len(train_loader.dataset)
                if val_loader is None:
                    print(f"[Epoch {epoch}] loss={avg_loss:.4f}")
                else:
                    val_acc = self._evaluate(val_loader, device)
                    print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    def _evaluate(self, loader: Iterable, device: str) -> float:
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                log_probs = self(xb)
                preds = log_probs.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total
