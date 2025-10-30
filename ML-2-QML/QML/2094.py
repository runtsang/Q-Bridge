"""Quantum branch for Quantum‑NAT using PennyLane variational circuit."""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QuantumNATEnhanced(nn.Module):
    """
    Quantum sub‑module that encodes a 16‑dim feature vector into a 4‑qubit circuit
    and returns a 4‑dim probability‑like vector.  The circuit uses
    angle‑encoding followed by two layers of trainable Ry/Rz rotations and
    entangling CNOTs.  The output is passed through a BatchNorm1d layer.

    The module is fully differentiable via PennyLane's autograd integration
    and can be attached to the classical backbone for end‑to‑end training.
    """
    def __init__(self,
                 n_wires: int = 4,
                 n_layers: int = 2,
                 device: str = 'default.qubit',
                 **kwargs) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.device = qml.device(device, wires=n_wires)
        self.theta = nn.Parameter(torch.randn(n_layers, n_wires, 2))
        self.norm = nn.BatchNorm1d(n_wires)

        @qml.qnode(self.device, interface='torch')
        def circuit(inputs: torch.Tensor):
            for i, w in enumerate(self.device.wires):
                qml.RY(inputs[:, i], wires=w)
            for l in range(n_layers):
                for q in range(n_wires):
                    qml.RY(self.theta[l, q, 0], wires=q)
                    qml.RZ(self.theta[l, q, 1], wires=q)
                for q in range(n_wires - 1):
                    qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] > self.n_wires:
            x = x[:, :self.n_wires]
        out = self.circuit(x)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
