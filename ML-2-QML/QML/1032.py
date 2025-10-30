"""Variational photonic fraud detection circuit using PennyLane."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import torch
from torch import nn


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


def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter approximation via two RX rotations
    qml.RX(params.bs_theta, wires=wires[0])
    qml.RX(params.bs_phi, wires=wires[1])

    # Phase rotations
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])

    # Squeezing gates approximated by RX (real part) and RZ (phase)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = r if not clip else _clip(r, 5)
        phi_val = phi
        qml.RX(r_val, wires=wires[i])
        qml.RZ(phi_val, wires=wires[i])

    # Second beam‑splitter
    qml.RX(params.bs_theta, wires=wires[0])
    qml.RX(params.bs_phi, wires=wires[1])

    # Displacement approximated by RZ rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_val = r if not clip else _clip(r, 5)
        phi_val = phi
        qml.RZ(r_val, wires=wires[i])
        qml.RZ(phi_val, wires=wires[i])

    # Kerr non‑linearity approximated by RZ rotation
    for i, k in enumerate(params.kerr):
        k_val = k if not clip else _clip(k, 1)
        qml.RZ(k_val, wires=wires[i])


def build_fraud_detection_qnode(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: qml.Device | None = None,
) -> qml.QNode:
    """Create a PennyLane QNode for the hybrid fraud detection model."""
    if device is None:
        device = qml.device("default.qubit", wires=2)

    @qml.qnode(device, interface="torch")
    def circuit() -> torch.Tensor:
        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)
        # Two measurement outputs
        return torch.stack(
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        )

    return circuit


class FraudDetectionHybrid(nn.Module):
    """
    Quantum‑classical hybrid fraud detection model.
    The quantum part is a PennyLane variational circuit; a classical linear head maps
    the two expectation values to a single output.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: qml.Device | None = None,
        post_linear: bool = True,
    ) -> None:
        super().__init__()
        self.qnode = build_fraud_detection_qnode(input_params, layers, device)
        if post_linear:
            self.linear = nn.Linear(2, 1)
        else:
            self.linear = None

    def forward(self, dummy: torch.Tensor | None = None) -> torch.Tensor:
        # The circuit does not depend on a classical input; dummy is kept for API compatibility
        out = self.qnode()
        if self.linear is not None:
            out = self.linear(out)
        return out


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_qnode",
    "FraudDetectionHybrid",
]
