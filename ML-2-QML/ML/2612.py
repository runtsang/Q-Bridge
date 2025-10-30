"""Hybrid classical‑quantum regressor combining a Pennylane feature extractor
with a fraud‑detection inspired classical head.

The model mirrors the EstimatorQNN feed‑forward network but replaces the
first layer with a quantum feature extractor.  Subsequent layers use the
same clipping, scaling and shift logic as the photonic fraud‑detection
example, ensuring that the classical head can learn from the quantum
features in a numerically stable way.
"""

from __future__ import annotations

import torch
from torch import nn
import pennylane as qml
from pennylane import numpy as np
from typing import Iterable, Sequence, Dict

# Quantum feature extractor ----------------------------------------------------
class QuantumFeatureExtractor(nn.Module):
    """A 2‑qubit feature extractor.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of variational layers.
    """

    def __init__(self, n_qubits: int = 2, n_layers: int = 1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameters for the circuit
        self.input_params = nn.Parameter(torch.randn(n_qubits))
        self.weight_params = nn.Parameter(torch.randn(n_layers * n_qubits))

        # Define the quantum node
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Encode inputs
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(self.weight_params[l * n_qubits + i], wires=i)
                # Entanglement
                if n_qubits > 1:
                    qml.CNOT(wires=[0, 1])
            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # inputs: (batch, n_qubits)
        return self.circuit(inputs)


# Classical head ---------------------------------------------------------------
class FraudLayer(nn.Module):
    """Classical layer inspired by the photonic fraud‑detection head."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)
        if clip:
            self.linear.weight.data.clamp_(-5.0, 5.0)
            self.linear.bias.data.clamp_(-5.0, 5.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(self.linear(inputs))
        return out * self.scale + self.shift


def _layer_from_params(
    bs_theta: float,
    bs_phi: float,
    phases: tuple[float, float],
    squeeze_r: tuple[float, float],
    squeeze_phi: tuple[float, float],
    displacement_r: tuple[float, float],
    displacement_phi: tuple[float, float],
    kerr: tuple[float, float],
    clip: bool = True,
) -> nn.Module:
    """Construct a FraudLayer with parameters derived from the photonic layer."""
    # Build weight matrix from beam‑splitter parameters
    weight = torch.tensor(
        [[bs_theta, bs_phi], [squeeze_r[0], squeeze_r[1]]], dtype=torch.float32
    )
    bias = torch.tensor(phases, dtype=torch.float32)
    scale = torch.tensor(displacement_r, dtype=torch.float32)
    shift = torch.tensor(displacement_phi, dtype=torch.float32)
    return FraudLayer(weight, bias, scale, shift, clip=clip)


def build_fraud_detection_head(
    input_params: Dict,
    layers: Iterable[Dict],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(**input_params, clip=False)]
    modules.extend(_layer_from_params(**layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# Hybrid estimator -------------------------------------------------------------
class EstimatorQNNHybrid(nn.Module):
    """Hybrid estimator that combines a quantum feature extractor with a
    fraud‑detection inspired classical head.
    """

    def __init__(
        self,
        quantum: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.quantum = quantum
        self.head = head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.quantum(inputs)
        return self.head(features)


def create_estimator_qnn_hybrid(
    n_qubits: int = 2,
    n_layers: int = 1,
    input_params: Dict | None = None,
    layers: Iterable[Dict] | None = None,
) -> EstimatorQNNHybrid:
    """Convenience factory returning a ready‑to‑train hybrid model."""
    quantum = QuantumFeatureExtractor(n_qubits, n_layers)
    head = build_fraud_detection_head(input_params or {}, layers or [])
    return EstimatorQNNHybrid(quantum, head)


__all__ = [
    "EstimatorQNNHybrid",
    "QuantumFeatureExtractor",
    "FraudLayer",
    "build_fraud_detection_head",
    "create_estimator_qnn_hybrid",
]
