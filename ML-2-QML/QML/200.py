"""Quantum hybrid model extending Quantum-NAT with amplitude‑encoded variational circuit."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np


class QuantumNATHybrid(nn.Module):
    """Amplitude‑encoded variational circuit + classical post‑processing."""

    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Pennylane device
        self.device = qml.device("default.qubit", wires=n_wires, shots=None)

        def circuit(inputs: np.ndarray, weights: np.ndarray) -> list[float]:
            # Encode first n_wires features as amplitudes
            qml.AmplitudeEmbedding(
                features=inputs, wires=range(n_wires), normalize=True
            )
            # Variational layers
            for layer in range(n_layers):
                for w in range(n_wires):
                    qml.RX(weights[layer, w, 0], wires=w)
                    qml.RY(weights[layer, w, 1], wires=w)
                    qml.RZ(weights[layer, w, 2], wires=w)
                # Entangling CZs
                for w in range(n_wires - 1):
                    qml.CZ(wires=[w, w + 1])
                qml.CZ(wires=[n_wires - 1, 0])
            # Measurement of Pauli‑Z on all wires
            return [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

        weight_shapes = {"weights": (n_layers, n_wires, 3)}
        self.qnode = qml.QNode(
            circuit,
            self.device,
            interface="torch",
            diff_method="backprop",
            weight_shapes=weight_shapes,
        )

        # Classical post‑processing
        self.fc = nn.Sequential(
            nn.Linear(n_wires, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Global average pooling to reduce spatial dimension
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)  # (bsz, 16)
        # Use first n_wires features for encoding
        features = pooled[:, : self.n_wires]
        # Run quantum circuit for each sample
        qout = torch.stack(
            [self.qnode(features[i]) for i in range(bsz)], dim=0
        )  # (bsz, n_wires)
        out = self.fc(qout)
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
