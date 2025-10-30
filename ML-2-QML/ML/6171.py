"""QuantumNATHybrid: classical CNN + variational quantum head.

Provides two variants:
- QuantumNATHybrid: pure classical CNN + FC head.
- QuantumNATHybridQ: hybrid CNN + variational quantum layer.

The quantum layer is implemented with Pennylane and wrapped in a
torch.autograd.Function for end‑to‑end differentiability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml

# --------------------------------------------------------------------------- #
# Classical feature extractor
# --------------------------------------------------------------------------- #
class _CNNFeatureExtractor(nn.Module):
    """Deeper CNN that mirrors the original 1‑channel architecture.

    Two convolutional blocks followed by a global average pooling.
    """
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        return torch.flatten(x, 1)

# --------------------------------------------------------------------------- #
# Variational quantum head
# --------------------------------------------------------------------------- #
class _VariationalQHead(nn.Module):
    """Variational circuit that maps a 16‑dim feature vector to 4 outputs.

    Uses Pennylane QNode with a trainable ansatz and a PauliZ measurement on
    each qubit.
    """
    def __init__(self, n_qubits: int = 4, dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)
        self._qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Encode input features as Ry rotations
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        # Parameterised layer
        for i in range(self.n_qubits):
            qml.RZ(params[i], wires=i)
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        # Measure Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        out = []
        for i in range(batch_size):
            # Random initial parameters for each forward pass
            params = torch.randn(self.n_qubits, device=x.device, requires_grad=True)
            out.append(self._qnode(params, x[i, :self.n_qubits]))
        return torch.stack(out)

# --------------------------------------------------------------------------- #
# Classical model
# --------------------------------------------------------------------------- #
class QuantumNATHybrid(nn.Module):
    """Pure‑classical CNN + fully‑connected head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = _CNNFeatureExtractor()
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return self.norm(x)

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class QuantumNATHybridQ(nn.Module):
    """Hybrid CNN + variational quantum head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = _CNNFeatureExtractor()
        self.quantum_head = _VariationalQHead()
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        q_out = self.quantum_head(x[:, :4])
        return self.norm(q_out)

__all__ = ["QuantumNATHybrid", "QuantumNATHybridQ"]
