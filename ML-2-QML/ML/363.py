"""QuantumClassifierModel__gen403: Classical feed‑forward classifier factory with optional regularisation.

The returned objects mirror the quantum helper interface so that the same experiment scripts
can operate on either backend.  The function accepts a depth and optional hyper‑parameters
for hidden width, dropout, and batch‑norm, and returns a ``torch.nn.Sequential`` model,
an encoding list, the sizes of each linear block, and the target class observables.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int | None = None,
    dropout: float | None = None,
    use_batchnorm: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier with optional regularisation layers.

    Parameters
    ----------
    num_features:
        Number of input features.
    depth:
        Number of hidden linear layers.
    hidden_dim:
        Width of each hidden layer.  Defaults to ``num_features``.
    dropout:
        Drop‑out probability after each hidden layer.  ``None`` disables dropout.
    use_batchnorm:
        Whether to insert a ``BatchNorm1d`` after each hidden linear layer.

    Returns
    -------
    network:
        ``nn.Sequential`` ready for training.
    encoding:
        ``Iterable[int]`` of feature indices that are linearly mapped into the network.
        For compatibility with the quantum helper this is simply ``range(num_features)``.
    weight_sizes:
        Number of trainable parameters per linear block (including bias).
    observables:
        ``List[int]`` representing the target class indices (``[0, 1]`` for binary classification).
    """
    hidden_dim = hidden_dim or num_features
    layers: List[nn.Module] = []

    in_dim = num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_dim)
        layers.append(linear)
        layers.append(nn.ReLU())
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if dropout:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    head = nn.Linear(in_dim, 2)
    layers.append(head)

    network = nn.Sequential(*layers)

    # Metadata for hybrid experiments
    encoding = list(range(num_features))
    weight_sizes = [m.weight.numel() + m.bias.numel() for m in network.modules() if isinstance(m, nn.Linear)]
    observables = [0, 1]

    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
