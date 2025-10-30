"""PennyLane implementation of the photonic fraud detection circuit.

The quantum circuit replaces the photonic gates with parameterised
single‑qubit rotations and entangling CZ gates, enabling hybrid
training on a simulator or hardware.  It follows the same layered
structure as the classical model, with optional clipping for stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pennylane as qml
from pennylane import numpy as np


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


class FraudDetection:
    """
    PennyLane implementation of the fraud detection circuit.

    Uses a variational circuit with parameterised RZ, RY, RX rotations
    and CZ entanglement.  The input vector is encoded as Pauli‑X
    rotations before the layer sequence.  The output is the expectation
    value of Pauli‑Z on the first qubit.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        n_qubits: int = 2,
    ) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params: List[FraudLayerParameters] = [input_params] + list(layers)
        self.circuit = qml.QNode(self._build_circuit, device=self.dev)

    def _apply_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        """Apply a single photonic layer using variational gates."""
        # Encode phases as RZ
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        # Encode squeezes as RY
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_enc = r if not clip else _clip(r, 5)
            qml.RY(r_enc, wires=i)

        # Encode displacements as RX
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_enc = r if not clip else _clip(r, 5)
            qml.RX(r_enc, wires=i)

        # Encode Kerr terms as CZ entanglement
        for i, k in enumerate(params.kerr):
            k_enc = k if not clip else _clip(k, 1)
            # Use the magnitude of k_enc to control a parametrised CZ
            qml.CZ(wires=[i, (i + 1) % self.n_qubits])

    def _build_circuit(self, *inputs: float) -> float:
        """Full circuit that processes the input and returns an expectation."""
        # Encode input as Pauli‑X rotations
        for i, val in enumerate(inputs):
            qml.PauliX(wires=i)

        # Apply each layer sequentially
        for idx, params in enumerate(self.params):
            clip = idx!= 0  # only clip subsequent layers
            self._apply_layer(params, clip)

        # Return expectation of Pauli‑Z on the first qubit
        return qml.expval(qml.PauliZ(0))

    def __call__(self, x: Sequence[float]) -> float:
        """Evaluate the circuit on a 2‑dimensional input vector."""
        return self.circuit(*x)


__all__ = ["FraudLayerParameters", "FraudDetection"]
