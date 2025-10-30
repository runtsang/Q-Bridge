"""Quantum version of the fraud‑detection model using PennyLane.

The circuit is a variational analogue of the photonic construction: each
layer is encoded as a sequence of single‑qubit rotations followed by a
controlled‑NOT gate that mimics the beam‑splitter entanglement.  The
parameters are reused from the classical definition, providing a
direct mapping between the two back‑ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, reused in the quantum circuit."""
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


def _apply_layer(qnode: Sequence[int], params: FraudLayerParameters, clip: bool) -> None:
    """Append the operations of a single layer to a PennyLane circuit."""
    for i, phase in enumerate(params.phases):
        qml.Rot(phase, 0, 0, wires=qnode[i])

    # Squeezing and displacement encoded as rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_eff = _clip(r, 5.0) if clip else r
        qml.Rot(r_eff, phi, 0, wires=qnode[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_eff = _clip(r, 5.0) if clip else r
        qml.Rot(r_eff, phi, 0, wires=qnode[i])

    # Beam‑splitter entanglement via a CNOT
    qml.CNOT(wires=[qnode[0], qnode[1]])

    # Kerr non‑linearity encoded as a Z‑rotation
    for i, k in enumerate(params.kerr):
        k_eff = _clip(k, 1.0) if clip else k
        qml.Rot(0, 0, k_eff, wires=qnode[i])


class FraudDetectionHybrid:
    """Hybrid quantum fraud‑detection model.

    The class mirrors the classical API but builds a PennyLane quantum node
    that can be evaluated on a classical input vector.  The circuit
    returns the sum of Pauli‑Z expectation values, which is used as the
    fraud‑score.
    """

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        self.input_params = input_params
        self.layers = list(layers)
        self.dev = qml.device("default.qubit", wires=2)
        self.qnode = qml.QNode(self._circuit, self.dev)

    @classmethod
    def from_parameters(cls, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> "FraudDetectionHybrid":
        """Convenience constructor that mimics the classical API."""
        return cls(input_params, layers)

    def _circuit(self, x0: float, x1: float) -> float:
        # Encode classical input as rotations
        qml.RX(x0, wires=0)
        qml.RX(x1, wires=1)

        _apply_layer([0, 1], self.input_params, clip=False)
        for layer in self.layers:
            _apply_layer([0, 1], layer, clip=True)

        # Output observable: sum of Pauli‑Z expectation values
        return qml.expval(qml.PauliZ(0)) + qml.expval(qml.PauliZ(1))

    def evaluate(self, inputs: np.ndarray) -> float:
        """Evaluate the quantum circuit on a 2‑dimensional input vector."""
        x0, x1 = inputs
        return self.qnode(x0, x1)

    def __call__(self, inputs: np.ndarray) -> float:
        return self.evaluate(inputs)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
