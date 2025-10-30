"""Quantum‑enhanced quanvolution filter using PennyLane."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumPatchEncoder(nn.Module):
    """Parameterized variational circuit that processes a 2×2 image patch."""

    def __init__(self, n_qubits: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        # Trainable parameters: shape (num_layers, n_qubits, 3)
        self.params = nn.Parameter(torch.randn(num_layers, n_qubits, 3))
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        self.qnode = qml.QNode(self._qcircuit, self.dev, interface="torch")

    def _qcircuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Encode each pixel with an RY gate
        for i in range(self.n_qubits):
            qml.RY(x[:, i], wires=i)
        # Variational layers
        for l in range(self.num_layers):
            for i in range(self.n_qubits):
                qml.RX(params[l, i, 0], wires=i)
                qml.RZ(params[l, i, 1], wires=i)
                qml.RY(params[l, i, 2], wires=i)
            # Entangling pattern
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        # Return expectation values of Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        return self.qnode(x, self.params)


class QuanvolutionFilter(nn.Module):
    """Hybrid quantum‑classical filter that applies the variational encoder to every 2×2 patch."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = QuantumPatchEncoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        # Reshape to (bsz, 28, 28)
        img = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = img[:, r : r + 2, c : c + 2]
                # Flatten patch to 4 values
                patch_flat = patch.reshape(bsz, 4)
                # Encode with quantum circuit
                out = self.encoder(patch_flat)
                patches.append(out)
        # Concatenate all patch outputs
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Full network that uses the quantum filter followed by a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
