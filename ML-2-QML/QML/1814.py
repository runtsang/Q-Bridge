"""
FraudDetectionModel – quantum variational circuit using PennyLane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single variational layer.

    The layout mirrors the photonic model but is interpreted as
    rotation angles and two‑qubit gate strengths.  ``bs_theta`` and
    ``bs_phi`` are used for RY and RZ rotations, ``phases`` for additional
    single‑qubit Z rotations, and the remaining tuples control the
    strength of controlled‑phase and controlled‑X operations.
    """
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(qnode: qml.QNode, params: FraudLayerParameters, clip: bool = False) -> None:
    """Append a single variational layer to the QNode circuit."""
    # Single‑qubit rotations
    qml.RY(params.bs_theta, wires=0)
    qml.RZ(params.bs_phi, wires=1)
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    # Two‑qubit entangling block
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_eff = r if not clip else _clip(r, 5.0)
        phi_eff = phi if not clip else _clip(phi, np.pi)
        qml.CRX(r_eff, wires=[i, (i + 1) % 2])
        qml.RZ(phi_eff, wires=i)

    # Displacement‑like rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_eff = r if not clip else _clip(r, 5.0)
        phi_eff = phi if not clip else _clip(phi, np.pi)
        qml.RY(r_eff, wires=i)
        qml.RZ(phi_eff, wires=i)

    # Kerr‑style phase shifts
    for i, k in enumerate(params.kerr):
        k_eff = k if not clip else _clip(k, 1.0)
        qml.RZ(k_eff, wires=i)


def build_fraud_detection_quantum_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device | None = None,
) -> qml.QNode:
    """Create a PennyLane variational QNode for fraud detection.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (non‑clipped) layer.
    layers : Iterable[FraudLayerParameters]
        Subsequent layers, each clipped to keep the circuit bounded.
    dev : qml.Device, optional
        Execution device; defaults to a default 2‑qubit qiskit simulator.

    Returns
    -------
    qml.QNode
        A quantum node that returns a single expectation value used as the
        fraud‑score.  The output is passed through a sigmoid to match the
        classical model's output range.
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # encode inputs into the first qubit
        qml.RY(inputs[0], wires=0)
        qml.RZ(inputs[1], wires=1)

        # first layer (no clipping)
        _apply_layer(qnode, input_params, clip=False)

        # subsequent layers (with clipping)
        for layer in params:
            _apply_layer(qnode, layer, clip=True)

        # measurement
        return qml.expval(qml.PauliZ(0))

    # Flatten the parameters into a trainable tensor
    param_list = [input_params] + list(layers)

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        # Convert all parameters to a torch tensor
        flat_params = torch.tensor(
            [
                [p.bs_theta, p.bs_phi, *p.phases,
                 *p.squeeze_r, *p.squeeze_phi,
                 *p.displacement_r, *p.displacement_phi,
                 *p.kerr] for p in param_list
            ],
            dtype=torch.float32,
        )
        raw = qnode(inputs, flat_params)
        return torch.sigmoid(raw)

    return forward

__all__ = ["FraudLayerParameters", "build_fraud_detection_quantum_model"]
