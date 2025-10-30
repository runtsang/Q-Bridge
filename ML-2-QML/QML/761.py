"""Quantum implementation using Pennylane with parameter‑shiftable entanglement."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

class QuantumNATEnhanced(nn.Module):
    """Quantum model with entanglement‑aware circuit and parameter‑shift training."""

    def __init__(self, num_qubits: int = 4, num_layers: int = 3, entanglement: str = "full", device: str = "default.qubit"):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.entanglement = entanglement
        self.dev = qml.device(device, wires=num_qubits)

        # Parameterized rotation angles for each layer
        self.params = nn.Parameter(torch.randn(num_layers, num_qubits, 3))

        # Batch norm for output
        self.norm = nn.BatchNorm1d(num_qubits)

        # Quantum node
        num_qubits = self.num_qubits
        entanglement = self.entanglement
        num_layers = self.num_layers
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, params):
            # Angle encoding of classical inputs
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Parameterized layers
            for l in range(num_layers):
                for q in range(num_qubits):
                    qml.RZ(params[l, q, 0], wires=q)
                    qml.RX(params[l, q, 1], wires=q)
                    qml.RY(params[l, q, 2], wires=q)
                # Entangling gates
                if entanglement == "full":
                    for i in range(num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif entanglement == "linear":
                    for i in range(num_qubits):
                        qml.CNOT(wires=[i, (i + 1) % num_qubits])
                # else: no entanglement

            # Return expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: expects input tensor of shape (batch, num_qubits).
        Outputs normalized expectation values of shape (batch, num_qubits).
        """
        bsz = x.shape[0]
        outputs = []
        for i in range(bsz):
            out = self.circuit(x[i], self.params)
            outputs.append(out)
        out = torch.stack(outputs, dim=0)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
