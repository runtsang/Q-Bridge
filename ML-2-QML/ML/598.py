"""Hybrid classical classifier with feature‑selection and adaptive regularisation.

The function `build_classifier_circuit` mirrors the original interface but
adds optional learnable feature masks and per‑layer regularisation
strengths.  The returned network is a `torch.nn.Module` that can be
trained end‑to‑end with autograd, while the returned metadata
(`encoding`, `weight_sizes`, `observables`) is kept for compatibility
with the quantum branch.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    learn_mask: bool = False,
    reg_strength: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier with optional learnable feature
    mask and adaptive regularisation.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int
        Number of hidden layers.
    learn_mask : bool, optional
        If ``True`` the input mask is a learnable parameter.
    reg_strength : float, optional
        Global regularisation strength that is attached to the model
        via the attribute ``.reg_strength``.  The model itself does not
        apply the penalty; it is expected to be added to the loss
        externally.

    Returns
    -------
    network : nn.Module
        The constructed network.  It contains an attribute
        ``.reg_strength`` for convenience.
    encoding : Iterable[int]
        Feature indices that are encoded by the network (identity).
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : list[int]
        Dummy list of class indices (kept for API compatibility).
    """
    class FeatureMask(nn.Module):
        def __init__(self, num_features: int, learn: bool):
            super().__init__()
            if learn:
                self.mask = nn.Parameter(torch.ones(num_features))
            else:
                self.register_buffer("mask", torch.ones(num_features))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.mask

    layers: list[nn.Module] = [FeatureMask(num_features, learn_mask)]
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    in_dim = num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    network.reg_strength = reg_strength

    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
