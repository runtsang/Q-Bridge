"""Hybrid classical classifier with depth‑aware regularization and optional dropout."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    reg_coeff: float = 0.0,
    dropout: float = 0.0,
    init: str = "he_normal",
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[float]]:
    """
    Construct a feed‑forward neural network that mirrors the structure of the
    quantum ansatz.

    Parameters
    ----------
    num_features : int
        Number of input features (also the width of every hidden layer).
    depth : int
        Number of hidden layers.
    reg_coeff : float, optional
        Coefficient for an L2 weight penalty.  The penalty is not applied
        automatically; it is returned in the ``observables`` list so that
        a training script can add it to the loss if desired.
    dropout : float, optional
        Dropout probability applied after every ReLU.  Set to ``0.0`` to
        disable dropout.
    init : str, optional
        Weight initialization scheme.  ``"he_normal"`` (default) uses Kaiming
        normal for ReLU activations; ``"xavier_uniform"`` is an alternative.

    Returns
    -------
    network : nn.Sequential
        The constructed classifier.
    encoding : Iterable[int]
        Indices of the input features used by the network (identity mapping).
    weight_sizes : Iterable[int]
        Number of learnable parameters in each layer (including the head).
    observables : List[float]
        A list containing the L2 regularization coefficient repeated for
        each layer.  The last element is ``reg_coeff`` so that the training
        script can easily sum them.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        if init == "he_normal":
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(linear.weight)
        layers.append(linear)
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)

    # The observables list carries the regularisation coefficient for each
    # layer; the last entry is the global coefficient.
    observables = [reg_coeff] * (len(weight_sizes) + 1)

    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
