"""Quanvolution model implemented with Pennylane variational circuits.

The quantum filter processes 2×2 patches of the input image.  Each patch
is encoded with Ry rotations, followed by a stack of learnable rotation
layers and a CNOT entangling pattern.  The measurement of Pauli‑Z on
each qubit yields a 4‑dimensional feature vector per patch.  The
resulting 4×14×14 feature map is flattened and passed through a linear
classifier, producing log‑softmax logits for classification.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class PennylaneQuanvolutionFilter(nn.Module):
    """Variational quanvolution filter using Pennylane."""
    def __init__(self, num_qubits: int = 4, num_layers: int = 3):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=None)
        # Learnable parameters for the rotation layers
        self.weights = nn.Parameter(torch.randn(num_layers, num_qubits, 3))

    def _circuit(self, features: torch.Tensor, weights: torch.Tensor):
        # Encode each pixel with an Ry rotation
        for i, f in enumerate(features):
            qml.RY(f.item(), wires=i)
        # Variational layers
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qml.Rot(*weights[layer, qubit], wires=qubit)
            # Entanglement pattern
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.num_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (B, 1, 28, 28)
        Output shape: (B, 4 * 14 * 14)
        """
        bsz = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2]          # (B, 2, 2)
                patch_flat = patch.view(bsz, -1) * math.pi
                batch_meas = []
                for i in range(bsz):
                    meas = qml.QNode(self._circuit, self.dev)(patch_flat[i], self.weights)
                    batch_meas.append(meas)
                batch_meas = torch.stack(batch_meas)
                patches.append(batch_meas)
        return torch.cat(patches, dim=1)

class QuanvolutionModel(nn.Module):
    """Hybrid quantum–classical classifier based on the Pennylane filter."""
    def __init__(self, num_qubits: int = 4, num_layers: int = 3, num_classes: int = 10):
        super().__init__()
        self.qfilter = PennylaneQuanvolutionFilter(num_qubits, num_layers)
        self.feature_dim = 4 * 14 * 14
        self.linear = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionModel"]
