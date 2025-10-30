"""Classical classifier class with extended functionality.

This class builds on the original build_classifier_circuit function
and adds:
- multi‑class support
- training utilities (optimizer, loss, early stopping)
- parameter count reporting
- flexible device selection (CPU/GPU)
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim


# Import the original circuit builder
from.QuantumClassifierModel import build_classifier_circuit


class QuantumClassifierModel:
    """Hybrid‑inspired classical classifier.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Depth of the feed‑forward network.
    num_classes : int, default 2
        Number of target classes.
    device : str or torch.device, default 'cpu'
        Device on which to run the model.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        num_classes: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.num_classes = num_classes
        self.device = torch.device(device)

        # Build network and metadata
        net, encoding, weight_sizes, observables = build_classifier_circuit(
            num_features, depth
        )
        self.network = net.to(self.device)
        self.encoding = encoding
        self.weight_sizes = weight_sizes
        self.observables = observables

        # Adjust final layer for arbitrary number of classes
        if self.num_classes!= 2:
            in_features = self.network[-1].in_features
            self.network[-1] = nn.Linear(in_features, self.num_classes)

    # ------------------------------------------------------------------
    # Forward and prediction utilities
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input ``x``."""
        return self.network(x.to(self.device))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class indices for input ``x``."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def train_one_epoch(
        self,
        data_loader: Iterable,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        clip_norm: float | None = None,
    ) -> float:
        """Train for one epoch and return average loss."""
        self.network.train()
        total_loss = 0.0
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            logits = self.forward(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if clip_norm is not None:
                nn.utils.clip_grad_norm_(self.network.parameters(), clip_norm)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        return total_loss / len(data_loader.dataset)

    def evaluate(self, data_loader: Iterable) -> Tuple[float, float]:
        """Return (accuracy, loss) on the validation set."""
        self.network.eval()
        correct, total, total_loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.forward(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
        accuracy = correct / total
        return accuracy, total_loss / total

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        """Return total number of learnable parameters."""
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_features={self.num_features}, "
            f"depth={self.depth}, num_classes={self.num_classes}, "
            f"params={self.num_parameters()})"
        )


__all__ = ["QuantumClassifierModel"]
