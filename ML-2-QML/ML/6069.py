"""UnifiedQCNN – Classical counterpart of the hybrid QCNN.

The module defines a single NN class that reproduces the layer structure of
the quantum circuit (feature map → conv → pool → conv → pool → conv → pool).
The weights are organised in a flat list so that the same ordering can be reused
for the quantum ansatz parameters.  This makes it trivial to initialise the
quantum model from classical weights or to compare their gradients.

The implementation uses PyTorch as the reference backend because it is
fully differentiable and can be trained with any of the standard optimisers.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, List, Tuple

# --------------------------------------------------------------------------- #
# 1.  Classical layer building blocks
# --------------------------------------------------------------------------- #
class _LinearTanh(nn.Module):
    """Linear → Tanh followed by optional scaling."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.linear(x))

# --------------------------------------------------------------------------- #
# 2.  Unified QCNN architecture
# --------------------------------------------------------------------------- #
class UnifiedQCNNModel(nn.Module):
    """
    Classical analogue of the Qiskit QCNN described in the reference pairs.

    Architecture:
        feature_map: 8 → 16
        conv1     : 16 → 16
        pool1     : 16 → 12
        conv2     : 12 → 12
        pool2     : 12 → 8
        conv3     : 16 → 8   (the second conv expands back to 16 for symmetry)
        pool3     : 8  → 4
        final     : 4  → 1

    The ordering of the weights is intentionally identical to the
    quantum ansatz parameters – see :func:`UnifiedQCNN.quantum_ansatz`.  The
    model is fully differentiable and can be exported to ONNX or TorchScript.
    """
    def __init__(self, feature_size: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.feature_map = _LinearTanh(feature_size, hidden)
        self.conv1     = _LinearTanh(hidden, hidden)
        self.pool1     = _LinearTanh(hidden, hidden - 4)
        self.conv2     = _LinearTanh(hidden - 4, hidden - 4)
        self.pool2     = _LinearTanh(hidden - 4, hidden - 8)
        self.conv3     = _LinearTanh(hidden - 8, hidden - 8)
        self.pool3     = _LinearTanh(hidden - 8, hidden // 2)
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2()
        x = self.conv3()
        return torch.sigmoid(self.head(x))

# --------------------------------------------------------------------------- #
# 3.  Helper for weight extraction / injection
# --------------------------------------------------------------------------- #
def _list_weights(model: nn.Module) -> List[torch.Tensor]:
    """Return a flat list of trainable weights in the same order as the
    quantum parameters – all weights are required for a gradient comparison."""
    return [w for w in model.parameters()]

def _inject_weights(model: nn.Module, weights: List[torch.Tensor]) -> None:
    """Load weights from a list into the model.  (The list length must match the
    weights *exactly* in order)."""
    for param, new in zip(model.parameters(), weights):
        with torch.no_grad():
            param.copy_(new)

# --------------------------------------------------------------------------- #
# 4.  Graph‑based diagnostics
# --------------------------------------------------------------------------- (see GraphQNN.py)
from.graph_utils import (
    feedforward as _feedforward,
    fidelity_adjacency as _fidelity_adjacency,
    random_network as _random_network,
    random_training_data as _random_training_data,
    state_fidelity as _state_fidelity,
)

# --------------------------------------------------------------------------- #
# 5.  Public API
# --------------------------------------------------------------------------- #
def UnifiedQCNN() -> UnifiedQCNNModel:
    """Convenience factory returning the default architecture."""
    return UnifiedQCNNModel()

__all__ = ["UnifiedQCNN", "UnifiedQCNNModel"]
