"""Hybrid quantum‑classical model for Quantum‑NAT using Pennylane."""

import torch
import torch.nn as nn
import pennylane as qml

class QFCModelGen100(nn.Module):
    """Hybrid model: classical conv + quantum variational circuit producing 4 features."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3, device: str = "cpu") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_wires)
        # Trainable parameters for the quantum circuit
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 2))
        self.norm = nn.BatchNorm1d(n_wires)

        # Classical feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # produce 4 features per sample
        )

        # Quantum node with torch interface
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Quantum circuit that encodes classical features and applies a variational ansatz."""
        # Encode classical features via RY rotations
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        # Variational layers
        for layer_idx in range(self.n_layers):
            for wire in range(self.n_wires):
                qml.RY(params[layer_idx, wire, 0], wires=wire)
                qml.RZ(params[layer_idx, wire, 1], wires=wire)
            # Entangling layer
            for wire in range(self.n_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        bsz = x.shape[0]
        features = self.feature_extractor(x).view(bsz, self.n_wires)
        # Quantum evaluation
        out = self.qnode(features, self.params)
        return self.norm(out)

__all__ = ["QFCModelGen100"]
