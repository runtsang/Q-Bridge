"""Quantum fraud‑detection circuit built with Pennylane.

The quantum module mirrors the classical photonic design by encoding the same set of
parameters into a variational circuit.  The circuit is parameterised by the
FraudLayerParameters dataclass, enabling side‑by‑side comparison and hybrid training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Parameter container – matches the classical counterpart.
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    num_qubits: int = 2


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _encode_features(qc: qml.QNode, params: FraudLayerParameters, clip: bool) -> None:
    """Apply a photonic‑style encoding to the qubits."""
    # Beam splitter equivalent – a rotation on two qubits
    for i in range(params.num_qubits):
        qml.RX(params.bs_theta, wires=i)
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RX(_clip(r, 5.0), wires=i)  # squeeze → rotation
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RY(_clip(r, 5.0), wires=i)  # displacement → rotation
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0), wires=i)  # Kerr → phase


# --------------------------------------------------------------------------- #
# Quantum circuit – a variational QNode that processes the encoded features
# --------------------------------------------------------------------------- #
class QuantumFraudDetector(nn.Module):
    """Classical‑to‑quantum hybrid model for fraud detection."""

    def __init__(self, num_qubits: int = 2, n_layers: int = 3):
        super().__init__()
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, num_qubits, 3))
        self.head = nn.Linear(num_qubits, 1)

    def _variational_layer(self, layer_idx: int) -> None:
        """A single variational layer with Ry‑Rz‑Ry entanglement."""
        for i in range(self.num_qubits):
            qml.RY(self.params[layer_idx, i, 0], wires=i)
            qml.RZ(self.params[layer_idx, i, 1], wires=i)
            qml.RY(self.params[layer_idx, i, 2], wires=i)
        # Entangle neighbouring qubits
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over a batch of feature vectors."""
        batch_size = x.shape[0]
        # Build a QNode that will be executed on the device
        @qml.qnode(self.dev, interface="torch")
        def circuit(features):
            for i in range(self.num_qubits):
                qml.RX(features[i], wires=i)
            for layer in range(self.n_layers):
                self._variational_layer(layer)
            return qml.expval(qml.PauliZ(wires=list(range(self.num_qubits))))

        # Encode each feature vector into the circuit
        outputs = torch.stack([circuit(x[i]) for i in range(batch_size)])
        return self.head(outputs).squeeze(-1)


# --------------------------------------------------------------------------- #
# Synthetic dataset – identical to the classical implementation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic dataset of superposition states with a sinusoid label."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class FraudDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic superposition data for quantum fraud detection."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": self.features[index],
            "target": self.labels[index],
        }


__all__ = [
    "FraudLayerParameters",
    "QuantumFraudDetector",
    "generate_superposition_data",
    "FraudDataset",
]
