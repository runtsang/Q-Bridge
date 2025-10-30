"""Hybrid QCNN for classical training with an optional quantum feature extractor.

The model emulates the convolution‑pooling structure of the original QCNN
while allowing a quantum feature map to be supplied as a callable.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple

class HybridQCNN(nn.Module):
    """Hybrid classical‑quantum QCNN.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    use_quantum : bool
        Whether a quantum extractor is active.
    quantum_extractor : Callable[[torch.Tensor], torch.Tensor] | None
        Function that maps a batch of input features to quantum features.
        It is expected to return a torch tensor compatible with the
        subsequent linear layers.
    """
    def __init__(self, num_features: int = 8,
                 use_quantum: bool = False,
                 quantum_extractor=None) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.quantum_extractor = quantum_extractor

        # Classical layers mirroring the original QCNN helper
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())
        self.conv1      = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1      = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2      = nn.Sequential(nn.Linear(12, 8),  nn.Tanh())
        self.pool2      = nn.Sequential(nn.Linear(8, 4),   nn.Tanh())
        self.conv3      = nn.Sequential(nn.Linear(4, 4),   nn.Tanh())
        self.head       = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.use_quantum and self.quantum_extractor is not None:
            x = self.quantum_extractor(x)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def set_quantum_extractor(self, extractor) -> None:
        """Register a callable quantum extractor."""
        self.quantum_extractor = extractor
        self.use_quantum = True

    def remove_quantum_extractor(self) -> None:
        """Disable the quantum extractor."""
        self.quantum_extractor = None
        self.use_quantum = False

def QCNN() -> HybridQCNN:
    """Factory returning the default HybridQCNN model."""
    return HybridQCNN()

# ----------------------------------------------------------------------
# Classical classifier helper mirroring the quantum interface
# ----------------------------------------------------------------------
def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Module,
                                                  Iterable[int],
                                                  Iterable[int],
                                                  list[int]]:
    """Return a feed‑forward classifier and metadata compatible with
    the quantum version defined in ``QuantumClassifierModel.py``."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["HybridQCNN", "QCNN", "build_classifier_circuit"]
