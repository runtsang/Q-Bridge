"""Classical classifier with optional residual connections and dropout, mirroring the quantum helper interface."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List

class QuantumClassifierModel(nn.Module):
    """
    Feed‑forward classifier that mimics the quantum API signature.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    depth : int
        Number of hidden layers.
    residual : bool, optional
        If True, add skip connections between every two layers.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """
    def __init__(self, num_features: int, depth: int, residual: bool = False, dropout: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.residual = residual
        self.dropout = dropout

        layers: List[nn.Module] = []
        in_dim = num_features

        for i in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            # Residual placeholder; actual addition handled in forward
            if residual and i % 2 == 1:
                layers.append(nn.Identity())
            in_dim = num_features

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional manual residuals."""
        if self.residual:
            out = x
            i = 0
            for module in self.body:
                if isinstance(module, nn.Identity):
                    out = out + out  # placeholder residual; can be replaced with real skip
                else:
                    out = module(out)
                i += 1
        else:
            out = self.body(x)
        return self.head(out)

    @property
    def encoding(self) -> Iterable[int]:
        """Return a trivial encoding (identity) to be compatible with the quantum signature."""
        return list(range(self.num_features))

    @property
    def weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per linear layer."""
        sizes = []
        for m in self.body:
            if isinstance(m, nn.Linear):
                sizes.append(m.weight.numel() + m.bias.numel())
        sizes.append(self.head.weight.numel() + self.head.bias.numel())
        return sizes

    @property
    def observables(self) -> List[int]:
        """Return indices of output logits."""
        return [0, 1]

__all__ = ["QuantumClassifierModel"]
