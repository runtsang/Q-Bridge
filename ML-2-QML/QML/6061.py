"""Hybrid fraud‑detection quantum circuit built with PennyLane.

This module mirrors the classical architecture but replaces the dense layers
with a variational photonic circuit.  It can be used as a QNode in a
hybrid training loop or evaluated on a simulator.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import pennylane as qml

# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
#  Helper to build a single photonic layer as a QNode
# --------------------------------------------------------------------------- #
def _build_layer(params: FraudLayerParameters, clip: bool = False) -> qml.QNode:
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor) -> torch.Tensor:
        # Encode classical inputs
        qml.RX(x[0], wires=0)
        qml.RX(x[1], wires=1)

        # Beam splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])

        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        # Squeezing (approximated for qubit device)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_eff = r if not clip else max(min(r, 5.0), -5.0)
            qml.RX(r_eff, wires=i)
            qml.RZ(phi, wires=i)

        # Displacement (approximated)
        for i, (d, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            d_eff = d if not clip else max(min(d, 5.0), -5.0)
            qml.RX(d_eff, wires=i)
            qml.RZ(phi, wires=i)

        # Kerr (approximated)
        for i, k in enumerate(params.kerr):
            k_eff = k if not clip else max(min(k, 1.0), -1.0)
            qml.RZ(k_eff, wires=i)

        return qml.expval(qml.PauliZ(0))

    return circuit

# --------------------------------------------------------------------------- #
#  Build the full hybrid fraud‑detection QNode
# --------------------------------------------------------------------------- #
def build_fraud_detection_qnode(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    clip: bool = True,
) -> qml.QNode:
    """
    Construct a hybrid QNode that first applies a classical linear
    embedding, then a sequence of photonic layers.
    """
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def hybrid_circuit(x: torch.Tensor) -> torch.Tensor:
        # Classical linear embedding (tanh activation)
        x = torch.tanh(nn.Linear(2, 2)(x))

        # First photonic layer (no clipping)
        x = _build_layer(input_params, clip=False)(x)

        # Subsequent layers (clipped)
        for params in layers:
            x = _build_layer(params, clip=clip)(x)

        # Final output
        return nn.Linear(1, 1)(x)

    return hybrid_circuit

__all__ = ["FraudLayerParameters", "build_fraud_detection_qnode"]
