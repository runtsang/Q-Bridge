"""QuantumNATEnhanced: Hybrid quantum‑classical model using PennyLane."""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml

class QuantumNATEnhanced(nn.Module):
    """Hybrid quantum‑classical model that extends the original Quantum‑NAT architecture."""
    def __init__(self, n_wires: int = 4, n_features: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_features = n_features
        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_wires)
        # Parameter vector for the variational circuit
        self.params = nn.Parameter(torch.randn(3 * self.n_wires))
        # Classical post‑processing
        self.postprocess = nn.Linear(self.n_wires, self.n_features)
        self.norm = nn.BatchNorm1d(self.n_features)

    def _circuit(self, params: torch.Tensor) -> torch.Tensor:
        # Reshape parameters
        params = params.view(3, self.n_wires)
        # Encode data into rotation angles
        for i in range(self.n_wires):
            qml.RX(self.data[:, i], wires=i)
        # Parameterized variational layer
        for i in range(self.n_wires):
            qml.RY(params[0, i], wires=i)
        for i in range(self.n_wires):
            qml.CNOT(wires=[i, (i+1) % self.n_wires])
        for i in range(self.n_wires):
            qml.RZ(params[1, i], wires=i)
        for i in range(self.n_wires):
            qml.CNOT(wires=[(i+1) % self.n_wires, i])
        for i in range(self.n_wires):
            qml.RX(params[2, i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Simple pooling to reduce dimensionality
        pooled = torch.nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        # Prepare data for encoding
        if pooled.shape[1] < self.n_wires:
            pad = torch.zeros(bsz, self.n_wires - pooled.shape[1], device=x.device)
            data = torch.cat([pooled, pad], dim=1)
        else:
            data = pooled[:, :self.n_wires]
        # Store data for the circuit
        self.data = data
        # Create a QNode
        qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        qout = qnode(self.params)
        out = self.postprocess(qout)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
