"""QuanvolutionPlus quantum filter using PennyLane."""

import torch
import torch.nn as nn
import pennylane as qml

class QuantumFilter(nn.Module):
    """Variational circuit acting on 4×4 image patches."""
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # trainable weights for the variational layers
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Encode inputs as Ry rotations
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)
            # Variational layers
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(*weights[l, i], wires=i)
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        patch_size = 4
        stride = 2
        outputs = []
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = x[:, :, i:i + patch_size, j:j + patch_size]
                patch_flat = patch.view(B, -1)
                # Normalize to [0, pi] for Ry encoding
                max_val = patch_flat.max()
                if max_val > 0:
                    patch_norm = torch.pi * patch_flat / max_val
                else:
                    patch_norm = torch.zeros_like(patch_flat)
                inputs = patch_norm[:, :self.n_qubits]
                out = self.circuit(inputs, self.weights)
                outputs.append(out)
        return torch.cat(outputs, dim=1)

class QuanvolutionPlusQuantumClassifier(nn.Module):
    """Quantum filter followed by a linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuantumFilter()
        dummy = torch.zeros(1, 1, 28, 28)
        feat = self.qfilter(dummy)
        self.linear = nn.Linear(feat.shape[1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return torch.log_softmax(logits, dim=-1)

__all__ = ["QuantumFilter", "QuanvolutionPlusQuantumClassifier"]
