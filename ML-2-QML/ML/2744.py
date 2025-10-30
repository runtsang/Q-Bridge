"""Hybrid classical‑quantum model that fuses Quantum‑NAT and GraphQNN.

The class `QuantumNATGraphQNN` can be used as a drop‑in replacement for the
original `QFCModel`.  It contains:
  * `FeatureExtractor` – a lightweight CNN that produces a 16‑dimensional
    feature vector.
  * `QuantumEncoder` – a quantum circuit that maps the pooled features to
    a 4‑qubit state and then a GraphQNNLayer that propagates the state
    through a sequence of random unitaries.
The forward pass returns a tuple `(classical_output, quantum_state)` where
the quantum state is a `torch.Tensor` of shape (batch, 4) after
measurement.  The quantum part also stores intermediate fidelities
between the state at each layer and the target unitary, which can
be used for training or analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum module defined in the QML counterpart.
# The quantum module shares the same class name but is defined in a
# separate file.  We alias it to avoid name clashes in the classical
# namespace.
try:
    from.QuantumNATGraphQNN_qml import QuantumNATGraphQNN as QuantumEncoder
except Exception as exc:
    raise ImportError(
        "QuantumNATGraphQNN_qml module could not be imported. "
        "Ensure the QML counterpart is present in the same package."
    ) from exc


class FeatureExtractor(nn.Module):
    """A compact CNN that mirrors the two‑layer Conv‑ReLU‑MaxPool
    architecture from the original Quantum‑NAT model.  The output
    random‑tied‑feature vector is *not* trainable beyond the
    encoder‑layer, and the subsequent quantum circuit will
    **treat** it as input.
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)  # shape (batch, 16)
        return x


class QuantumNATGraphQNN(nn.Module):
    """Hybrid classical‑quantum model that fuses Quantum‑NAT’s convolution‑to‑quantum mapping
    with GraphQNN’s graph‑based fidelity analysis.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.quantum_encoder = QuantumEncoder(n_qubits=n_qubits, n_layers=n_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], object]:
        """Return quantum measurement, per‑layer fidelities, and fidelity graph."""
        features = self.feature_extractor(x)
        out, fidelities, graph = self.quantum_encoder(features)
        return out, fidelities, graph


__all__ = ["QuantumNATGraphQNN"]
