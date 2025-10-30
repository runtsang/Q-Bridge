"""Quantum‑NAT enhanced model using Pennylane variational circuit."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


class QuantumNATEnhanced(nn.Module):
    """Hybrid model: classical encoder + variational quantum circuit + classical head."""
    def __init__(self, num_classes: int = 4, n_wires: int = 4, device: str = "default.qubit"):
        super().__init__()
        self.n_wires = n_wires
        self.device = device

        # Variational circuit with two parameter layers and CNOT entanglement
        def circuit(params, x):
            # Encode classical data as rotation angles
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # First parameterized layer
            for i in range(self.n_wires):
                qml.RX(params[0, i], wires=i)
            # Entangling CNOTs
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Second parameterized layer
            for i in range(self.n_wires):
                qml.RZ(params[1, i], wires=i)
            # Measure all qubits in Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        self.qnode = qml.QNode(circuit, qml.device(device, wires=n_wires))
        # Learnable parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(2, n_wires))
        # Classical post‑processing head
        self.postprocess = nn.Sequential(
            nn.Linear(n_wires, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_wires) with values in [-π, π]
        Returns:
            Tensor of shape (batch, num_classes)
        """
        batch = x.shape[0]
        # Run the QNode for each sample in the batch
        out = torch.stack([torch.tensor(self.qnode(self.params, x[i].detach().cpu().numpy())) for i in range(batch)], dim=0)
        out = self.postprocess(out)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
