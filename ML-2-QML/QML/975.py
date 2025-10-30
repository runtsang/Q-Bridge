"""Quantum fraud detection model using a variational photonic circuit.

The circuit mirrors the classical architecture but replaces linear layers
with parameterised Gaussian gates.  The shared class name `FraudDetectionHybrid`
provides a straight‑forward interface for hybrid training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import torch
import torch.nn as nn

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    qml.BSgate(params.bs_theta, params.bs_phi, wires=modes)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=modes[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5), phi, wires=modes[i])
    qml.BSgate(params.bs_theta, params.bs_phi, wires=modes)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=modes[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5), phi, wires=modes[i])
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1), wires=modes[i])

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    device: str = "default.qubit",
    shots: int = 1000,
) -> qml.QNode:
    """Create a Pennylane QNode for the hybrid fraud detection model."""
    dev = qml.device(device, wires=2, shots=shots)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor) -> torch.Tensor:
        # The circuit is defined for a single sample; batch processing
        # can be achieved by looping or vectorisation.
        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)
        # Measure expectation values of Pauli‑Z on both modes
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return circuit

class FraudDetectionHybrid:
    """Quantum fraud detection model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers.
    device : str, optional
        Pennylane device name.
    shots : int, optional
        Number of measurement shots.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        device: str = "default.qubit",
        shots: int = 1000,
    ) -> None:
        self.circuit = build_fraud_detection_program(
            input_params,
            layers,
            device=device,
            shots=shots,
        )
        # Classical post‑processing layer
        self.post = nn.Linear(2, 1)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, 2)
        z_vals = self.circuit(inputs)
        # z_vals is a tuple of two tensors shape (batch,)
        z_stack = torch.stack(z_vals, dim=-1)  # shape (batch, 2)
        return self.post(z_stack)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
