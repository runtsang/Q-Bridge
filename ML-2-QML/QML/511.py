"""Quantum-enhanced fraud detection model using PennyLane.

This module extends the original photonic circuit by:
- Translating the classical parameters into a variational ansatz.
- Adding an entanglement layer that can be swapped between CNOT and CSWAP.
- Supporting hybrid training with automatic differentiation via PennyLane.
- Providing a `FraudDetectionHybrid` class that exposes a `forward` method
  compatible with PyTorch backends.

The design keeps the same parameter semantics while enabling quantum
circuit simulation and gradient computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import pennylane as qml
from pennylane import numpy as np
import torch

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
    entangler: str = "cnot"  # new hyperparameter

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(
    q: Sequence,
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    # Simple rotation equivalent to a beam‑splitter
    qml.Rgate(params.bs_theta, params.bs_phi, wires=q[0])
    qml.Rgate(params.bs_theta, params.bs_phi, wires=q[1])
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=q[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5), phi, wires=q[i])
    # Entanglement
    if params.entangler == "cnot":
        qml.CNOT(wires=[q[0], q[1]])
    else:
        qml.CSWAP(wires=[q[0], q[1]])
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=q[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5), phi, wires=q[i])
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1), wires=q[i])

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device,
) -> qml.QNode:
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor) -> torch.Tensor:
        # Encode 2‑D input as rotations
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)
        return qml.expval(qml.PauliZ(0))
    return circuit

class FraudDetectionHybrid:
    """Hybrid quantum‑classical fraud detection model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers.
    dev_name : str, optional
        PennyLane device name (default: 'default.qubit').
    wires : int, optional
        Number of qubits (default: 2).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dev_name: str = "default.qubit",
        wires: int = 2,
    ) -> None:
        self.dev = qml.device(dev_name, wires=wires)
        self.circuit = build_fraud_detection_program(input_params, layers, self.dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the expectation value for the given 2‑D input."""
        return self.circuit(x)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
