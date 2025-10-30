"""Quantum quanvolutional filter using a parameterised PennyLane circuit."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionFilter(nn.Module):
    """
    Parameterised quantum filter that maps 2x2 image patches to a 4‑dimensional feature vector.
    The circuit consists of Ry encodings followed by two entangling layers with trainable parameters.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # PennyLane device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Trainable parameters for the entangling layers
        self.params = nn.Parameter(torch.randn(n_layers, 2))
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor):
            # Encode inputs into qubits using Ry rotations
            for i in range(self.n_qubits):
                qml.RY(inputs[:, i], wires=i)
            # Entangling layers
            for idx in range(self.n_layers):
                qml.CNOT(wires=[idx, (idx + 1) % self.n_qubits])
                qml.RY(params[idx, 0], wires=idx)
                qml.CNOT(wires=[idx, (idx + 1) % self.n_qubits])
                qml.RY(params[idx, 1], wires=idx)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        Returns a (batch, 4*14*14) feature tensor.
        """
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r+2, c:c+2]  # (batch, 2, 2)
                patch = patch.reshape(bsz, -1)  # (batch, 4)
                # Run the circuit in batch mode
                measurement = torch.stack(self._circuit(patch, self.params), dim=1)
                patches.append(measurement)
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid network that applies the quantum filter and a classical linear head.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
