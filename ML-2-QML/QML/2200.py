"""Variational circuit implementing the fraud‑detection logic with PennyLane.

The photonic operations (beam‑splitter, squeezing, displacement, Kerr)
are emulated by qubit rotations and controlled gates.  The circuit
is fully differentiable, making it suitable for hybrid training
with PyTorch or TensorFlow back‑ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pennylane as qml
from pennylane import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters that mimic a photonic layer."""

    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class FraudDetection:
    """PennyLane implementation of the fraud‑detection circuit.

    The mapping is intentionally simple:
        • Beam‑splitter ⟶ pair of controlled‑RX gates.
        • Squeezing ⟶ RX + RZ rotations.
        • Displacement ⟶ RY + RZ rotations.
        • Kerr ⟶ RZ rotation (bounded to |k| ≤ 1).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dev: qml.Device | None = None,
        clip_bound: float = 5.0,
    ) -> None:
        self.dev = dev or qml.device("default.qubit", wires=2)
        self.clip_bound = clip_bound
        self.input_params = input_params
        self.layers = list(layers)

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        """Apply the photonic‑like operations to two qubits."""
        # Beam‑splitter emulation
        qml.CRX(params.bs_theta, wires=0)
        qml.CRX(params.bs_phi, wires=1)

        # Phase rotations
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        # Squeezing (RX + RZ)
        for i, (r, p) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_clipped = self._clip(r, self.clip_bound) if clip else r
            qml.RX(r_clipped, wires=i)
            qml.RZ(p, wires=i)

        # Displacement (RY + RZ)
        for i, (r, p) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_clipped = self._clip(r, self.clip_bound) if clip else r
            qml.RY(r_clipped, wires=i)
            qml.RZ(p, wires=i)

        # Kerr (RZ, bounded to |k| ≤ 1)
        for i, k in enumerate(params.kerr):
            k_clipped = self._clip(k, 1.0) if clip else k
            qml.RZ(k_clipped, wires=i)

    def circuit(self, *trainable: np.ndarray) -> np.ndarray:
        """PennyLane QNode that returns the expectation value
        of Pauli‑Z on the joint state of qubits 0 and 1.
        """
        self._apply_layer(self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(layer, clip=True)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def qnode(self) -> qml.QNode:
        """Return a compiled QNode for use in optimisation."""
        return qml.QNode(self.circuit, self.dev)

    @classmethod
    def build_fraud_detection_circuit(
        cls,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dev: qml.Device | None = None,
        clip_bound: float = 5.0,
    ) -> "FraudDetection":
        """Convenience constructor for the circuit."""
        return cls(input_params, layers, dev=dev, clip_bound=clip_bound)

    def expectation(self, params: List[float]) -> float:
        """Evaluate the circuit with a list of trainable parameters."""
        qnode = self.qnode()
        return qnode(*params)


__all__ = ["FraudLayerParameters", "FraudDetection"]
