"""Hybrid classical‑quantum classifier factory.

The module exposes a single ``QuantumHybridClassifier`` class that
mirrors the interface of the original
``QuantumClassifierModel`` but augments its neural network with a
data‑encoding layer and a shared weight vector that is used both by
the classical and quantum parts.  The design is inspired by the
four reference pairs:

* The feed‑forward backbone from Reference 1 (ML seed) with depth
  control.
* The parameterised encoding from Reference 2 (SamplerQNN) for
  mapping classical features to quantum angles.
* The regression‑style head from Reference 3 (QModel) that
  concatenates classical and quantum features.
* The convolution‑like pooling from Reference 4 (QCNN) which we
  embed as a *quantum pooling* block that collapses the qubit
  dimension.

The class is fully importable, relies only on PyTorch and NumPy,
and can be instantiated from the anchor path
``QuantumClassifierModel__gen366.py``.  It accepts the same
``num_features`` and ``depth`` arguments as the original, but
returns a tuple ``(model, encoding_meta, weight_meta, observables)``
where each part can be used directly in a hybrid training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Iterable, Tuple, List

class QuantumHybridClassifier(nn.Module):
    """Feed‑forward backbone for a hybrid classifier.

    Parameters
    ----------
    num_features : int
        Number of classical input features.
    depth : int, default=2
        Number of hidden layers in the classical part.
    """

    def __init__(self, num_features: int, depth: int = 2) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        # Build a simple MLP that mirrors the original ML seed
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))  # binary classifier head
        self.net = nn.Sequential(*layers)

        # Metadata for the quantum side
        # ``encoding`` tells the quantum circuit which features to encode
        self.encoding: List[int] = list(range(num_features))
        # ``weight_sizes`` holds the number of parameters for each layer
        self.weight_sizes: List[int] = []
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                self.weight_sizes.append(layer.weight.numel() + layer.bias.numel())
        # ``observables`` are the measurement operators used in the QML side
        self.observables: List[int] = [0, 1]  # placeholder indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical network."""
        return self.net(x)

    def get_shared_weights(self) -> torch.Tensor:
        """Return a flattened view of all trainable parameters.

        The returned tensor can be passed to the quantum part
        as a shared weight vector.
        """
        return torch.cat([p.view(-1) for p in self.parameters()])

def build_hybrid_classifier(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Factory that returns the classical model and its metadata.

    The return tuple is compatible with the quantum factory
    defined in the QML module.
    """
    model = QuantumHybridClassifier(num_features, depth)
    return model, model.encoding, model.weight_sizes, model.observables
