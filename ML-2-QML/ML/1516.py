"""Robust classical classifier factory for the QuantumClassifierModel project.

The :class:`QuantumClassifierModel` class mirrors the original API but adds
advanced layers and introspection utilities.  It can be instantiated
directly or used via the :meth:`build_classifier_circuit` classmethod,
which returns the network together with metadata required for hybrid
experiments.

Typical usage::

    from ml_module import QuantumClassifierModel
    net, enc, wsize, obs = QuantumClassifierModel.build_classifier_circuit(
        num_features=20, depth=3)

    # Forward pass
    logits = net(torch.randn(1, 20))

The returned tuple matches the signature expected by downstream
training scripts and hybrid wrappers.

The class is intentionally lightweight – it only imports :mod:`torch`
and does not depend on any external training utilities, making it
easy to plug into existing pipelines or to extend further.

"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """Shared classical classifier class.

    The class offers a static factory that produces a feed‑forward
    neural network with configurable depth and regularisation.
    """

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        dropout_rate: float = 0.1,
    ) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
        """
        Construct a feed‑forward classifier and metadata similar to the
        quantum variant.

        Parameters
        ----------
        num_features:
            Dimensionality of the input feature vector.
        depth:
            Number of hidden layers.
        dropout_rate:
            Dropout probability applied after each hidden layer.

        Returns
        -------
        network:
            ``torch.nn.Sequential`` model ready for training.
        encoding:
            List of input feature indices (identity encoding).
        weight_sizes:
            Number of trainable parameters per linear layer.
        observables:
            Class indices for the classification heads.
        """
        layers: List[nn.Module] = []
        in_dim = num_features

        # Identity encoding: record the feature indices
        encoding = list(range(num_features))

        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))

            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)

        observables = [0, 1]  # class indices

        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
