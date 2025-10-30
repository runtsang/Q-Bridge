"""Classical hybrid model that optionally delegates to a quantum module.

This module implements a convolution‑style network that mirrors the
QCNNModel and extends it with the ability to plug in a quantum
sub‑module.  The quantum sub‑module is provided externally via the
``quantum_layer`` callable.  This design keeps the core network
trainable with standard PyTorch optimisers while allowing the
back‑propagation to flow through an attached quantum circuit.

The class name ``QCNNHybrid`` is shared with the quantum implementation
in the QML module to keep the API consistent across the two
representations.
"""

import torch
from torch import nn
from typing import Optional, Callable

class QCNNHybrid(nn.Module):
    """
    Classical hybrid QCNN with an optional quantum sub‑module.

    Parameters
    ----------
    in_features : int
        Number of input features (must match the feature map size).
    hidden_features : int
        Width of the fully‑connected layers.
    out_features : int
        Size of the prediction output.
    quantum_layer : Optional[Callable[[torch.Tensor], torch.Tensor]]
        A callable that takes a batch of feature vectors and returns
        a tensor of the same shape.  If ``None`` the network is fully
        classical.
    """
    def __init__(
        self,
        in_features: int = 8,
        hidden_features: int = 16,
        out_features: int = 1,
        quantum_layer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        # Classical feature‑map.
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
        )

        # Sequence of mixed layers: alternating classical FC and
        # placeholders for quantum blocks.
        self.fc1 = nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(hidden_features, hidden_features // 2), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(hidden_features // 2, hidden_features // 2), nn.Tanh())

        self.head = nn.Linear(hidden_features // 2, out_features)

        # Optional quantum layer; it is executed after the classical
        # encoder and before the final head.
        self.quantum_layer = quantum_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        The input ``x`` should be of shape (batch, in_features).
        """
        x = self.feature_map(x)
        x = self.fc1(x)
        # Quantum block, if provided.
        if self.quantum_layer is not None:
            x = self.quantum_layer(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.head(x)

def QCNN() -> QCNNHybrid:
    """Factory returning a default ``QCNNHybrid`` instance."""
    return QCNNHybrid()
