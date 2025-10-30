"""Hybrid classical layer that unifies a fully‑connected unit and a deep classifier.

The class can operate in two modes:
  * ``fcl`` – a single linear layer with tanh activation, emulating a quantum fully‑connected layer.
  * ``classifier`` – a multi‑layer feed‑forward network with ReLU activations and a 2‑class head.

Both modes expose a `run` method that accepts a flat list of parameters and returns a NumPy array
containing the forward pass output.  The `weight_sizes` helper reports the number of trainable
parameters in each linear block, mirroring the quantum observables list.

The design follows the *scaling_paradigm: combination* directive, allowing the classical
implementation to be used as a drop‑in replacement for the quantum version in benchmarks or
hybrid training loops.
"""

from __future__ import annotations

from typing import Iterable, List, Union

import numpy as np
import torch
from torch import nn


class HybridLayer(nn.Module):
    """
    Classical hybrid layer supporting a fully‑connected unit or a deep classifier.

    Parameters
    ----------
    num_features : int, default=1
        Number of input features / qubits.
    depth : int, default=1
        Depth of the classifier network (ignored in ``fcl`` mode).
    mode : str, {"fcl", "classifier"}, default="fcl"
        Operation mode.  ``"fcl"`` yields a single linear layer; ``"classifier"``
        builds a network with `depth` hidden layers and a 2‑output head.
    """

    def __init__(self, num_features: int = 1, depth: int = 1, mode: str = "fcl") -> None:
        super().__init__()
        self.mode = mode
        self.num_features = num_features

        if mode == "fcl":
            self.linear = nn.Linear(num_features, 1)
        elif mode == "classifier":
            layers: List[nn.Module] = []
            in_dim = num_features
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, num_features))
                layers.append(nn.ReLU())
                in_dim = num_features
            layers.append(nn.Linear(in_dim, 2))
            self.network = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported mode {mode!r}. Choose 'fcl' or 'classifier'.")

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the layer with the given parameters.

        The flat list ``thetas`` must match the number of trainable parameters in the chosen mode.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32)

        if self.mode == "fcl":
            theta_tensor = theta_tensor.view(-1, 1)
            out = torch.tanh(self.linear(theta_tensor)).mean(dim=0)
            return out.detach().numpy()
        else:  # classifier
            out = self.network(theta_tensor)
            return out.detach().numpy()

    def weight_sizes(self) -> List[int]:
        """Return the number of parameters per linear block."""
        sizes: List[int] = []
        if self.mode == "fcl":
            sizes.append(self.linear.weight.numel() + self.linear.bias.numel())
        else:
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    sizes.append(layer.weight.numel() + layer.bias.numel())
        return sizes

    def observables(self) -> List[int]:
        """
        Stub to mirror the quantum interface.

        For the classical model we simply return a placeholder list of observable indices.
        """
        return list(range(len(self.weight_sizes())))


__all__ = ["HybridLayer"]
