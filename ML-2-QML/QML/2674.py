"""Hybrid quantum-classical estimator using a Pennylane variational circuit for feature extraction.

The module defines a single class `HybridEstimatorQNN` that:
- extracts 2×2 patches from a 28×28 grayscale image,
- encodes each patch into a 4‑qubit state via Ry rotations,
- applies a two‑layer variational circuit,
- measures the expectation of Pauli‑Z on each qubit,
- flattens all patch measurements into a feature vector,
- feeds the vector into a linear regressor.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = qml.device("default.qubit", wires=4)

def _quantum_layer(patch: torch.Tensor) -> list[float]:
    """Variational circuit that maps a 4‑element patch to a 4‑dimensional feature vector."""
    for i in range(4):
        qml.RY(patch[i] * torch.pi / 2, wires=i)
    # Two‑layer variational block
    for i in range(4):
        qml.RY(0.1, wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

qnode = qml.QNode(_quantum_layer, dev, interface="torch")

class QuanvolutionFilter(nn.Module):
    """Quantum filter that uses a Pennylane variational circuit on image patches."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2].reshape(bsz, -1)
                patch_feats = []
                for i in range(bsz):
                    patch_feats.append(qnode(patch[i]))
                patch_feats = torch.stack(patch_feats, dim=0)
                features.append(patch_feats)
        return torch.cat(features, dim=1)

class HybridEstimatorQNN(nn.Module):
    """Hybrid regressor that combines the quantum filter with a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        return self.linear(features)

__all__ = ["HybridEstimatorQNN"]
