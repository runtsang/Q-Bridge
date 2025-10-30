"""Enhanced fully connected layer with training support and batch processing."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from typing import Iterable, Sequence

class FCL(nn.Module):
    """
    Fully connected layer that accepts a sequence of input features and
    returns the mean of a tanh activation.  The class exposes a ``run``
    method that mirrors the original API but now supports batch input
    and optional weight updates via autograd.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_outputs : int, default 1
        Number of linear outputs.
    """
    def __init__(self, n_features: int = 1, n_outputs: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer followed by tanh activation.
        """
        return torch.tanh(self.linear(x))

    def run(self, thetas: Iterable[float], batch: bool = False) -> np.ndarray:
        """
        Mimic the original ``run`` signature while allowing batched input.

        Parameters
        ----------
        thetas : Iterable[float]
            Input values or weight vector.  If ``batch`` is ``False``,
            ``thetas`` is interpreted as a single sample; otherwise it
            must be an iterable of iterables.
        batch : bool, default False
            Whether the input is batched.

        Returns
        -------
        np.ndarray
            Mean activation over the processed samples.
        """
        if batch:
            values = torch.as_tensor(list(thetas), dtype=torch.float32)
        else:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        output = self.forward(values)
        return output.mean(dim=0).detach().numpy()

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        lr: float = 1e-3,
        loss_fn: nn.Module = nn.MSELoss(),
    ) -> float:
        """
        Perform a single gradient‑descent step.

        Parameters
        ----------
        data : torch.Tensor
            Input batch.
        target : torch.Tensor
            Ground‑truth targets.
        lr : float
            Learning rate.
        loss_fn : nn.Module
            Loss function.

        Returns
        -------
        float
            Loss value for the step.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        pred = self.forward(data)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()

__all__ = ["FCL"]
