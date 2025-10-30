"""Unified classical classifier with optional quantum augmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Optional


class UnifiedClassifier(nn.Module):
    """
    Classical feed‑forward network that can be used as a drop‑in replacement
    for the original QuantumClassifierModel.  The architecture is
    modular: a list of hidden layers, optional gating, and a linear
    output head.  The design is inspired by the classical seed and
    the gate‑based LSTM logic from the quantum seed.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    use_gating : bool, optional
        If True, a sigmoid gate is applied after each hidden layer.
        The gate is a small feed‑forward network that learns to
        re‑weight the activations.  This mirrors the gate logic in
        the quantum LSTM implementation.
    gating_hidden : int, optional
        Hidden size of the gating network.  Default equals
        ``num_features``.
    device : str or torch.device, optional
        Device for tensors.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        use_gating: bool = False,
        gating_hidden: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_gating = use_gating
        self.device = device

        self.layers: nn.ModuleList = nn.ModuleList()
        self.weight_sizes: List[int] = []
        in_dim = num_features

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            self.layers.append(linear)
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
            if use_gating:
                gate = nn.Sequential(
                    nn.Linear(num_features, gating_hidden or num_features),
                    nn.Sigmoid(),
                )
                self.layers.append(gate)

        self.head = nn.Linear(in_dim, 2)
        self.weight_sizes.append(self.head.weight.numel() + self.head.bias.numel())
        self.layers.append(self.head)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Log‑probabilities of shape (batch, 2).
        """
        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                gate = layer
                x = gate(x) * x
            else:
                x = layer(x)
                x = F.relu(x) if layer!= self.head else x
        return F.log_softmax(x, dim=1)

    # ------------------------------------------------------------------
    #  Auxiliary helpers –  mirror the interface of the quantum seed
    # ------------------------------------------------------------------
    def get_encoding_indices(self) -> List[int]:
        """Return the indices that would be encoded in a quantum circuit."""
        return list(range(self.num_features))

    def get_weight_sizes(self) -> List[int]:
        """Return the list of trainable parameter counts per layer."""
        return self.weight_sizes

    def get_observables(self) -> List[int]:
        """Return a placeholder list of observables (classical equivalent)."""
        return [0, 1]  # two-class output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_features={self.num_features}, "
            f"depth={self.depth}, use_gating={self.use_gating})"
        )


__all__ = ["UnifiedClassifier"]
