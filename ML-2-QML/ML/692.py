"""Classical Quanvolution module with residual connections and dropout."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quanvolution(nn.Module):
    """
    Classical hybrid filter that mimics a quantum convolution.
    The architecture consists of a 2×2 convolution followed by a residual
    1×1 convolution, batch‑normalisation, a linear head and an optional
    dropout.  The class exposes a :meth:`train_step` helper that can be
    used in a standard PyTorch training loop.
    """

    def __init__(self, dropout_prob: float = 0.3) -> None:
        super().__init__()
        # 2×2 convolution reduces 28×28 to 14×14
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Residual 1×1 convolution keeps spatial resolution
        self.conv2 = nn.Conv2d(4, 4, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(4 * 14 * 14, 10)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)
        out += residual
        out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.dropout(out)
        return self.logsoftmax(out)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        """
        Execute one training step.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, 1, 28, 28).
        y : torch.Tensor
            Target labels of shape (B,).
        optimizer : torch.optim.Optimizer
            Optimiser that will update all learnable parameters.
        loss_fn : nn.Module
            Loss function that accepts logits and targets.

        Returns
        -------
        float
            The scalar loss value.
        """
        self.train()
        optimizer.zero_grad()
        logits = self.forward(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["Quanvolution"]
