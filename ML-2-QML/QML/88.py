"""Quantum implementation of the fraud‑detection circuit using PennyLane.

The circuit is a variational photonic‑inspired ansatz that can be
differentiated automatically and fused with a classical read‑out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (used only for
    documentation; the quantum circuit maps them to gate parameters)."""
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


def _apply_layer(wires: Tuple[int, int], params: FraudLayerParameters, clip: bool) -> None:
    """Append gates to the current QNode to emulate a photonic layer."""
    # Beam splitter (BS) – approximated as a rotation between two qubits
    theta = params.bs_theta
    phi = params.bs_phi
    qml.RX(theta, wires=wires[0])
    qml.RX(theta, wires=wires[1])
    qml.CNOT(wires=wires)

    # Phase shifters
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])

    # Squeezing and displacement – approximated with parameterized rotations
    for i, (r, phi_r) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = _clip(r, 5.0) if clip else r
        qml.RY(r_val, wires=wires[i])
        qml.RZ(phi_r, wires=wires[i])

    # Displacement
    for i, (r, phi_r) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_val = _clip(r, 5.0) if clip else r
        qml.RY(r_val, wires=wires[i])
        qml.RZ(phi_r, wires=wires[i])

    # Kerr nonlinearity – approximated with a ZZ rotation
    for i, k in enumerate(params.kerr):
        k_val = _clip(k, 1.0) if clip else k
        qml.PhaseShift(k_val, wires=wires[i])


def build_fraud_detection_qprog(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev=None,
) -> qml.QNode:
    """Return a PennyLane QNode implementing the fraud‑detection ansatz."""
    if dev is None:
        dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(x: np.ndarray) -> np.ndarray:
        # Encode the two‑dimensional input into the qubits
        qml.BasisState(x, wires=range(2))
        _apply_layer((0, 1), input_params, clip=False)
        for params in layers:
            _apply_layer((0, 1), params, clip=True)
        # Expectation value of PauliZ on the first qubit as the output
        return qml.expval(qml.PauliZ(0))

    return circuit


__all__ = ["FraudLayerParameters", "build_fraud_detection_qprog"]
