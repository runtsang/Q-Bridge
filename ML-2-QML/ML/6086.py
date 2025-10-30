"""Classical implementation of the HybridClassifier.

The classical head is a fully‑connected network that can output binary
or multi‑class probabilities.  It replaces the quantum expectation
layer in the original model and can be used as a drop‑in
replacement for experiments that do not require a quantum backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class HybridClassifier(nn.Module):
    """
    Classical hybrid classifier with a flexible head.

    Parameters
    ----------
    in_features : int
        Number of features from the preceding network.
    hidden_dims : int | tuple[int,...]
        Size(s) of hidden layers in the head.  If an int is provided,
        a single hidden layer is created.
    num_classes : int
        Number of output classes.  For binary classification set
        ``num_classes=1`` and a sigmoid activation is applied.
    mode : Literal["binary", "multiclass"]
        Determines the activation used in the final layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: int | tuple[int,...] = 120,
        num_classes: int = 1,
        mode: Literal["binary", "multiclass"] = "binary",
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        layers = []
        prev_dim = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*layers)

        self.mode = mode
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        if self.mode == "binary":
            probs = torch.sigmoid(logits)
            return torch.cat((probs, 1 - probs), dim=-1)
        elif self.mode == "multiclass":
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            raise ValueError(f"Unsupported mode {self.mode!r}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.head[0].in_features}, "
            f"hidden_dims={tuple(layer.out_features for layer in self.head if isinstance(layer, nn.Linear))[1:-1]}, "
            f"num_classes={self.num_classes}, mode={self.mode})"
        )


__all__ = ["HybridClassifier"]
