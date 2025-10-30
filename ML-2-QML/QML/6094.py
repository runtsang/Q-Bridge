"""Quantum‑enhanced model for Quantum‑NAT experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QFCModelExtended(nn.Module):
    """Variational quantum layer inspired by Quantum‑NAT."""

    def __init__(self, n_qubits: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        # Linear encoder to map pooled features to qubit states
        self.encoder = nn.Linear(16, n_qubits)
        # Variational parameters
        self.q_params = nn.Parameter(torch.randn(num_layers, n_qubits))
        # Quantum device
        self.device = qml.device("default.qubit", wires=n_qubits)
        # Batch normalization on the quantum output
        self.norm = nn.BatchNorm1d(n_qubits)

        # Define the quantum circuit
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            for layer in range(self.num_layers):
                for q in range(self.n_qubits):
                    qml.RY(inputs[q], wires=q)
                    qml.RZ(params[layer, q], wires=q)
                # Entanglement layer
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(wires=q)) for q in range(self.n_qubits)]

        self.circuit = qml.qnode(circuit, device=self.device, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        encoded = self.encoder(pooled)
        out = []
        for i in range(bsz):
            out.append(self.circuit(encoded[i], self.q_params))
        out = torch.stack(out)
        return self.norm(out)


__all__ = ["QFCModelExtended"]
