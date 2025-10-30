"""Enhanced classical fully‑connected model with modular design and a manual weight loader.

The original seed exposed a single linear layer wrapped in a `run` method that
accepted a list of scalars.  This upgrade replaces that single layer with an
arbitrary‑depth feed‑forward network.  The network supports optional dropout,
batch‑normalisation and a `set_parameters` routine that accepts a flat list of
weights and biases – the same format the quantum seed used for its rotation
angles.  This keeps the public API identical while adding expressive power.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
import numpy as np


class FCLModel(nn.Module):
    """
    Feed‑forward network with optional dropout and batch‑norm.
    """
    def __init__(
        self,
        n_features: int = 1,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or []

        layers: List[nn.Module] = []
        in_dim = n_features

        # Build hidden layers
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------
    def set_parameters(self, thetas: Iterable[float]) -> None:
        """
        Load a flat list of parameters into the network.

        The list is interpreted in the same order as `torch.nn.Linear` weight
        and bias tensors concatenated in a single vector.  This mirrors the
        quantum seed’s `run` signature and allows a side‑by‑side comparison
        between classical and quantum models.
        """
        flat = torch.tensor(list(thetas), dtype=torch.float32)
        pointer = 0
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                w_shape = module.weight.shape
                b_shape = module.bias.shape
                w_size = w_shape.numel()
                b_size = b_shape.numel()

                w = flat[pointer : pointer + w_size].view(w_shape)
                pointer += w_size
                b = flat[pointer : pointer + b_size].view(b_shape)
                pointer += b_size

                module.weight.data.copy_(w)
                module.bias.data.copy_(b)

        if pointer!= flat.numel():
            raise ValueError(
                f"Parameter list length {flat.numel()} does not match "
                f"network size {pointer}."
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    # ------------------------------------------------------------------
    # Run interface (keeps compatibility with the seed)
    # ------------------------------------------------------------------
    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the network on a single input vector defined by `thetas`.

        The parameters are first loaded into the network, then the network
        processes a one‑dimensional tensor of shape `(1, n_features)`.
        The output is returned as a NumPy array of shape `(1,)`.
        """
        self.set_parameters(thetas)
        # The seed used a single feature; we mimic that behaviour.
        # The caller can provide more features by extending the parameter list.
        with torch.no_grad():
            output = self.forward(torch.tensor(
                [list(thetas)], dtype=torch.float32
            ))
        return output.squeeze().detach().numpy()


__all__ = ["FCLModel"]
