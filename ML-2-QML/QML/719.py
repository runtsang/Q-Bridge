"""Hybrid classical‑quantum model using Pennylane variational circuit for Quantum‑NAT."""

import pennylane as qml
import torch
import torch.nn as nn

class QFCModel(nn.Module):
    """Classical encoder followed by a Pennylane variational circuit."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers

        # Classical feature extractor
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten(),
            nn.Linear(16, n_wires)
        )

        # Pennylane quantum device and QNode
        self.dev = qml.device("default.qubit", wires=n_wires, shots=1024)
        self.variational_params = nn.Parameter(torch.randn(n_layers, n_wires, 3))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(params: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            # Encode classical features onto qubits
            for i in range(self.n_wires):
                qml.RX(features[i], wires=i)
            # Parameterized variational layers
            for layer in range(self.n_layers):
                for w in range(self.n_wires):
                    qml.RX(params[layer, w, 0], wires=w)
                    qml.RY(params[layer, w, 1], wires=w)
                    qml.RZ(params[layer, w, 2], wires=w)
                # Simple entangling pattern
                for w in range(self.n_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(wires=w)) for w in range(self.n_wires)]

        self.qnode = _circuit
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        encoded = self.encoder(x)  # shape: (bsz, n_wires)
        # Run the quantum circuit for each batch element
        out = []
        for i in range(bsz):
            out.append(self.qnode(self.variational_params, encoded[i]))
        out = torch.stack(out, dim=0)
        return self.norm(out)

__all__ = ["QFCModel"]
