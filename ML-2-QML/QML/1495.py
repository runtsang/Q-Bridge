"""Quantum hybrid binary classifier using Pennylane."""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# Device used for all quantum circuits
device = qml.device("default.qubit", wires=2)


@qml.qnode(device, interface="torch")
def quantum_circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Two‑qubit variational circuit that encodes the input as Ry rotations,
    entangles the qubits with a CNOT, then measures the expectation of Z on qubit 0.
    """
    for i in range(2):
        qml.RY(weights[i] * x[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0))


class HybridQuantumHead(nn.Module):
    """Hybrid head that evaluates the variational quantum circuit."""
    def __init__(self, n_qubits: int = 2, init_weights: torch.Tensor | None = None):
        super().__init__()
        self.n_qubits = n_qubits
        if init_weights is None:
            init_weights = torch.randn(n_qubits)
        self.weights = nn.Parameter(init_weights.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, dim = x.shape
        assert dim == self.n_qubits, f"Expected {self.n_qubits} features per sample"
        x_reshaped = x.view(batch, self.n_qubits)
        # Compute expectation for each sample in the batch
        expvals = torch.stack([quantum_circuit(x_reshaped[i], self.weights)
                               for i in range(batch)])
        probs = torch.sigmoid(expvals.unsqueeze(-1))
        return torch.cat([probs, 1 - probs], dim=-1)


class HybridBinaryNet(nn.Module):
    """CNN backbone followed by a hybrid quantum head."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.head = HybridQuantumHead(n_qubits=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)


__all__ = ["HybridBinaryNet"]
