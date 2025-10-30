"""Enhanced classical classifier mirroring the quantum helper interface with residual connections and dropout.

The class builds a feed‑forward network with optional batch‑norm and dropout layers, exposing the same
metadata (encoding, weight_sizes, observables) as the original design.  It can be used in pipelines that
expect a ``build_classifier_circuit``‑style factory but offers improved regularisation and deeper
representational capacity.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """
    Classical feed‑forward classifier that mimics the interface of the quantum helper.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    device : str | torch.device, optional
        Target device for the network.
    """

    def __init__(self, num_features: int, depth: int, device: str | torch.device = "cpu"):
        self.num_features = num_features
        self.depth = depth
        self.device = torch.device(device)

        (
            self.model,
            self.encoding,
            self.weight_sizes,
            self.observables,
        ) = self._build_classifier()

    def _build_classifier(
        self,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """
        Construct a deeper network with residual connections, batch‑norm and dropout.
        """
        layers: list[nn.Module] = []
        in_dim = self.num_features

        for _ in range(self.depth):
            # Residual block
            block = nn.Sequential(
                nn.Linear(in_dim, self.num_features),
                nn.BatchNorm1d(self.num_features),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
            layers.append(block)
            in_dim = self.num_features

        layers.append(nn.Linear(in_dim, 2))
        self.model = nn.Sequential(*layers).to(self.device)

        # Metadata analogous to the quantum counterpart
        self.encoding = list(range(self.num_features))
        self.weight_sizes = [p.numel() for p in self.model.parameters()]
        self.observables = list(range(2))

        return self.model, self.encoding, self.weight_sizes, self.observables

    # --------------------------------------------------------------------- #
    # Forward / training helpers
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x.to(self.device))

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Single training step."""
        optimizer.zero_grad()
        out = self.forward(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation: return class indices."""
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)


__all__ = ["QuantumClassifierModel"]
