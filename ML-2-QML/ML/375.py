from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout: float = 0.1
) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a robust classical classifier that mimics the interface of the
    quantum helper.  The network is built from a stack of residual blocks
    (linear → ReLU → dropout) followed by a linear head.  The returned
    metadata mirrors the quantum version:

    * ``encoding`` – indices of the input features.
    * ``weight_sizes`` – number of trainable parameters per layer.
    * ``observables`` – symbolic output labels (here simply ``[0, 1]``).

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of residual blocks.
    dropout : float, default 0.1
        Drop‑out probability applied after each ReLU.

    Returns
    -------
    nn.Module
        The constructed classifier.
    Iterable[int]
        List of feature indices used for encoding.
    List[int]
        Parameter counts for each layer in the network.
    List[int]
        Dummy observable identifiers for downstream compatibility.
    """
    layers: List[nn.Module] = []
    in_dim = num_features

    # Build residual blocks
    for _ in range(depth):
        block = nn.Sequential(
            nn.Linear(in_dim, num_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        layers.append(block)
        in_dim = num_features

    # Final linear head
    head = nn.Linear(in_dim, 2)
    layers.append(head)

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = list(range(2))

    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
