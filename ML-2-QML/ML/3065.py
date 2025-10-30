"""Hybrid fraud‑detection architecture combining quantum and classical layers.

The module exposes two main classes:
  * :class:`QuantumFraudLayer` – a PennyLane qnode that maps a 2‑dimensional input
    vector to two expectation values (〈Z〉 on each qubit).
  * :class:`FraudDetectionHybrid` – a PyTorch module that feeds the quantum
    outputs into a classical sequential network built from
    :func:`build_fraud_detection_program`.

The design keeps the original clipping behaviour of the classical layers
while allowing the quantum part to learn its own parameters during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn

# ----------------------------------------------------------------------
# Quantum feature extractor
# ----------------------------------------------------------------------
@dataclass
class QuantumFraudLayerParameters:
    """Parameters for the PennyLane quantum circuit."""
    r1: float = 0.0  # rotation around X for qubit 0
    r2: float = 0.0  # rotation around X for qubit 1
    entangle_strength: float = 0.0  # parameter for RZZ gate

class QuantumFraudLayer(nn.Module):
    """A 2‑qubit PennyLane qnode that returns two expectation values."""

    def __init__(self, device: str = "default.qubit", shots: int = 1024) -> None:
        super().__init__()
        self.device = device
        self.shots = shots
        self.qnode = qml.QNode(self._circuit, qml.device(device, wires=2, shots=shots))

    def _circuit(self, r1: float, r2: float, entangle_strength: float) -> Sequence[float]:
        qml.RX(r1, wires=0)
        qml.RX(r2, wires=1)
        qml.RZZ(entangle_strength, wires=[0, 1])
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor of shape (batch, 2)
            Each row contains the parameters (r1, r2) for the quantum circuit.
            The entanglement strength is fixed at 0.0 for simplicity.

        Returns
        -------
        torch.Tensor of shape (batch, 2)
            Expectation values from the two qubits.
        """
        # Convert to numpy for the qnode; the qnode accepts scalars, not tensors.
        batch = inputs.cpu().numpy()
        outputs = []
        for r1, r2 in batch:
            expvals = self.qnode(r1, r2, 0.0)
            outputs.append(expvals)
        return torch.tensor(outputs, dtype=torch.float32, device=inputs.device)

# ----------------------------------------------------------------------
# Classical network construction
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Hybrid model
# ----------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that first extracts quantum features
    and then processes them with a classical network.

    Parameters
    ----------
    quantum_layer : nn.Module
        A module that maps a 2‑dimensional input to a 2‑dimensional output
        (e.g. :class:`QuantumFraudLayer`).
    classical_params : Iterable[FraudLayerParameters]
        Parameters for the classical layers. The first element is used as
        the input layer; the rest are hidden layers.

    Notes
    -----
    The quantum layer is expected to output a tensor of shape (batch, 2).
    The classical network expects the same shape, so the hybrid model
    can be trained end‑to‑end with back‑propagation through the qnode
    (via PennyLane's autograd support).
    """

    def __init__(
        self,
        quantum_layer: nn.Module,
        classical_params: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.quantum = quantum_layer
        self.classical = build_fraud_detection_program(*classical_params)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantum_features = self.quantum(inputs)
        return self.classical(quantum_features)

__all__ = [
    "QuantumFraudLayer",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
