"""Quantum‑enhanced model for Quantum‑NAT.

The implementation keeps the same public interface as the classical
module but replaces the fully‑connected head with a parameter‑efficient
variational quantum circuit.  The circuit is built with PennyLane
and trained end‑to‑end together with the classical backbone.

The model can be used as a drop‑in replacement in any training loop
that expects a `torch.nn.Module` with a 4‑class output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumCircuit(nn.Module):
    """Parameter‑efficient variational circuit operating on 4 qubits."""
    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        # Learnable rotation angles for each depth layer and qubit
        self.params = nn.Parameter(torch.randn(depth, n_wires))
        # Exact simulator – no shots required for gradients
        self.dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(self.dev, interface='torch')
        def _circuit(features: torch.Tensor) -> torch.Tensor:
            """
            features: Tensor of shape (batch, n_wires)
            Returns expectation values of Pauli‑Z on each qubit
            as a tensor of shape (batch, n_wires).
            """
            # Feature map: encode each feature into a single qubit
            for i in range(n_wires):
                qml.RY(features[:, i], wires=i)
            # Variational ansatz
            for d in range(depth):
                for i in range(n_wires):
                    qml.RY(self.params[d, i], wires=i)
                # Entangling layer (cyclic CNOTs)
                for i in range(n_wires):
                    qml.CNOT(wires=[i, (i + 1) % n_wires])
            return torch.stack(
                [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)],
                dim=1,
            )

        self.circuit = _circuit

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.circuit(features)


class QuantumNATEnhanced(nn.Module):
    """Hybrid classical‑quantum model mirroring the classical API."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Classical encoder identical to the seed
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.quantum_block = QuantumCircuit(n_wires=self.n_wires, depth=2)
        # Linear mapping from 4‑dimensional quantum output to 4 classes
        self.fc = nn.Linear(4, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.encoder(x)
        # Reduce to a 4‑dimensional vector for the quantum circuit
        pooled = F.avg_pool2d(feats, kernel_size=7).view(bsz, 16)
        grouped = pooled.view(bsz, 4, 4).mean(dim=2)  # (batch, 4)
        q_out = self.quantum_block(grouped)
        logits = self.fc(q_out)
        return self.norm(logits)


__all__ = ["QuantumNATEnhanced"]
