"""Quantum‑classical quanvolution filter using PennyLane.

The filter processes each 2×2 patch of a 28×28 image through a
parameterised variational circuit on 4 qubits.  The circuit consists
of input encoding with Ry gates, a trainable RZ layer, and a fixed
entangling pattern.  The expectation values of Pauli‑Z operators
are returned as a 4‑dimensional feature vector per patch.  The
module is fully differentiable via PennyLane’s interface with
PyTorch and can be trained on a GPU when the 'default.qubit' device
is used with the 'qiskit' backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple

# PennyLane device; supports GPU if the backend is enabled
device = qml.device("default.qubit", wires=4, shots=1)

def _quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    PennyLane QNode that encodes a 4‑element input vector and applies a
    single layer of trainable RZ gates followed by a fixed CNOT chain.
    Returns the expectation values of Pauli‑Z on all wires.
    """
    for i in range(4):
        qml.RY(inputs[i], wires=i)
    # trainable RZ layer
    for i, w in enumerate(weights):
        qml.RZ(w, wires=i % 4)
    # entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Wrap the QNode for PyTorch interface
quantum_circuit = qml.QNode(_quantum_circuit, interface="torch", device=device)


class QuanvolutionFilter(nn.Module):
    """
    Quantum 2×2 patch extractor.  The filter iterates over every
    non‑overlapping 2×2 patch of the input image, runs it through a
    parameterised variational circuit, and concatenates the resulting
    4‑dimensional feature vectors.
    """
    def __init__(self, num_qubits: int = 4, num_params: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.weight = nn.Parameter(torch.randn(num_params))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch, 1, 28, 28)
        returns: tensor of shape (batch, 4 * 14 * 14)
        """
        bsz = x.size(0)
        img = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = img[:, r:r + 2, c:c + 2]
                patch = patch.view(bsz, -1)  # (batch, 4)
                out = torch.stack([quantum_circuit(patch[i], self.weight)
                                   for i in range(bsz)], dim=0)
                patches.append(out)
        # concatenate all patch outputs
        out = torch.cat(patches, dim=1)
        if self.dropout.p > 0.0:
            out = self.dropout(out)
        return out


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid neural network that uses the quantum quanvolution filter
    followed by a linear head.  The design is compatible with the
    original API while providing a true quantum contribution.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(dropout=dropout)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
