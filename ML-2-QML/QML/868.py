"""Hybrid quantum‑classical quanvolution network.

The new class `QuanvolutionGen139` implements a variational
quantum filter that operates on 2x2 image patches.  The filter
uses a trainable ansatz with depth `depth` and measures the
expectation values of Pauli‑Z on each wire.  The output
features are concatenated and fed to a linear head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionGen139(nn.Module):
    """Variational quantum filter + linear head."""
    def __init__(self,
                 depth: int = 2,
                 num_wires: int = 4,
                 shots: int = 0,
                 device: str = "default.qubit",
                 num_classes: int = 10):
        super().__init__()
        self.depth = depth
        self.num_wires = num_wires
        self.shots = shots
        self.q_device = qml.device(device, wires=num_wires, shots=shots)
        # Trainable parameters for the ansatz
        self.params = nn.Parameter(torch.randn(depth, num_wires, 3))
        # Linear head
        self.linear = nn.Linear(num_wires * 14 * 14, num_classes)

        @qml.qnode(self.q_device, interface="torch", diff_method="backprop")
        def quantum_layer(input_vec, params):
            # Encode the 4‑element input vector via Ry rotations
            for i in range(num_wires):
                qml.RY(input_vec[i], wires=i)
            # Variational ansatz
            for d in range(depth):
                for w in range(num_wires):
                    qml.RX(params[d, w, 0], wires=w)
                    qml.RY(params[d, w, 1], wires=w)
                    qml.RZ(params[d, w, 2], wires=w)
                # Entangling layer
                for w in range(num_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[num_wires - 1, 0])
            # Measure expectation values of PauliZ on each wire
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.quantum_layer = quantum_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        bsz = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2]  # shape (batch, 2, 2)
                patch = patch.view(bsz, -1)  # shape (batch, 4)
                # Apply quantum layer
                q_out = self.quantum_layer(patch, self.params)
                patches.append(q_out)
        # Concatenate all patches: shape (batch, 4*14*14)
        features = torch.cat(patches, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen139"]
