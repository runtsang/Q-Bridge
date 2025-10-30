"""
FCL__gen260.py – Classical multi‑layer fully‑connected layer with training support.
"""

from __future__ import annotations
from typing import Iterable, Sequence, List

import numpy as np
import torch
from torch import nn, optim


def FCL() -> "FCL":
    """
    Return a trainable, multi‑layer perceptron that mimics a fully‑connected quantum layer.
    The returned object implements ``run`` and ``train`` methods.
    """
    class FCL(nn.Module):
        """
        Multi‑layer perceptron with two hidden layers and dropout.
        """

        def __init__(self, n_features: int = 1, hidden_dim: int = 16) -> None:
            super().__init__()
            # Two hidden layers with ReLU non‑linearity
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, 1),
            )
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
            self.loss_fn = nn.MSELoss()

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """
            Forward pass: compute the mean activation over the provided parameters.
            ``thetas`` is an iterable of float scalars.
            """
            inputs = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            with torch.no_grad():
                output = self.net(inputs).mean(dim=0)
            return output.detach().cpu().numpy()

        def train_step(self, thetas: Iterable[float], targets: Iterable[float]) -> float:
            """
            Single gradient descent step.
            Returns the loss value.
            """
            self.optimizer.zero_grad()
            inputs = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            preds = self.net(inputs).squeeze()
            loss = self.loss_fn(preds, torch.tensor(list(targets), dtype=torch.float32))
            loss.backward()
            self.optimizer.step()
            return loss.item()

    return FCL()
__all__ = ["FCL"]
