"""Quantum quanvolution network using a parameterized PennyLane circuit."""
from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """
    Quantum‑classical hybrid network: a parameterised 2×2 quantum kernel applied to each
    image patch followed by a linear classifier.  The quantum kernel is a variational
    circuit with learnable rotation angles and CNOT entangling layers.  The circuit
    is executed on a PennyLane autograd device so gradients flow through the quantum
    part.
    """
    def __init__(self,
                 num_qubits: int = 4,
                 num_layers: int = 2,
                 num_classes: int = 10,
                 device: str = "default.qubit.autograd") -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=self.num_qubits)
        # Learnable circuit parameters: shape (num_layers, num_qubits)
        self.params = nn.Parameter(torch.randn(num_layers, num_qubits))
        self.classifier = nn.Linear(num_qubits * 14 * 14, num_classes)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor):
            # inputs: (batch, num_qubits)
            for i in range(self.num_qubits):
                qml.RY(inputs[:, i], wires=i)
            for l in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.RY(params[l, i], wires=i)
                # Entangling layer
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.num_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 28, 28)
        B = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2]          # (B, 2, 2)
                patch = patch.view(B, -1)              # (B, 4)
                patches.append(patch)
        # Stack patches: (B, 14*14, 4)
        patches = torch.stack(patches, dim=1)
        # Reshape to (B*14*14, 4) for batch processing
        flat_patches = patches.view(-1, self.num_qubits)
        q_outputs = self.circuit(flat_patches, self.params)  # (B*14*14, 4)
        # Reshape back
        q_features = q_outputs.view(B, 14 * 14, self.num_qubits)
        # Flatten for classifier
        flat_features = q_features.view(B, -1)
        logits = self.classifier(flat_features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
