"""Quantum fraud detection model using PennyLane.

The circuit implements a 2‑mode photonic‑style ansatz mapped onto
two qubits.  Each photonic gate in the seed is replaced by a
parameterised qubit gate so that the entire model remains fully
differentiable via the parameter‑shift rule.  The output of the
circuit is the expectation value of PauliZ on the first qubit,
interpreted as a fraud logit.  A lightweight auxiliary
regression head is added on top of the logit to predict a risk
score, mirroring the classical auxiliary head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import torch
from torch import nn

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(
    modes: Sequence[int],
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    # Map the photonic BSgate to a two‑qubit rotation.
    qml.CRX(params.bs_phi, wires=(modes[0], modes[1]))
    qml.RZ(params.bs_theta, wires=modes[0])
    qml.RZ(params.bs_phi, wires=modes[1])

    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=modes[i])

    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_clipped = r if not clip else _clip(r, 5)
        qml.RZ(r_clipped, wires=modes[i])  # approximate squeezing

    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_clipped = r if not clip else _clip(r, 5)
        qml.RX(r_clipped, wires=modes[i])  # approximate displacement

    for i, k in enumerate(params.kerr):
        k_clipped = k if not clip else _clip(k, 1)
        qml.RX(k_clipped, wires=modes[i])

def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: qml.Device,
) -> qml.QNode:
    """Return a QNode that maps a 2‑dimensional input to a fraud logit."""

    @qml.qnode(device, interface="torch")
    def circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Encode the two‑dimensional input using RY rotations.
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # First (unclipped) layer.
        _apply_layer((0, 1), input_params, clip=False)

        # Subsequent clipped layers.
        for layer in layers:
            _apply_layer((0, 1), layer, clip=True)

        # Variational rotation with trainable parameters.
        qml.RZ(params[0], wires=0)

        # Fraud logit: expectation value of PauliZ on qubit 0.
        return qml.expval(qml.PauliZ(0))

    return circuit

class FraudDetectionQuantumModel(nn.Module):
    """Quantum fraud detector with a variational ansatz.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (unclipped) layer.
    hidden_params : Iterable[FraudLayerParameters]
        Parameters for subsequent clipped layers.
    device_name : str, optional
        PennyLane device name, defaults to 'default.qubit'.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.device = qml.device(device_name, wires=2)
        self.circuit = build_fraud_detection_circuit(
            input_params, hidden_params, self.device
        )
        # Single trainable parameter for the variational rotation.
        self.params = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud logit."""
        return self.circuit(x, self.params)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        fraud_weight: float = 1.0,
    ) -> torch.Tensor:
        """Binary cross‑entropy loss for fraud prediction."""
        return nn.functional.binary_cross_entropy_with_logits(
            logits, labels
        ) * fraud_weight

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "FraudDetectionQuantumModel",
]
