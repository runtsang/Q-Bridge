"""
Quantum fraud detection model using a PennyLane variational circuit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


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


class FraudDetectionCircuit:
    """Variational circuit that emulates the layered photonic structure."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        self.dev = qml.device("default.qubit", wires=2)
        self.params = [input_params] + list(layers)
        self.vqc = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, *weights, **kwargs):
        # weights is a flattened list of all trainable parameters
        idx = 0
        for layer in self.params:
            # beam splitter equivalent: two‑qubit entangling rotation
            theta, phi = layer.bs_theta, layer.bs_phi
            qml.CRX(theta, wires=0)
            qml.CRY(phi, wires=1)

            # single‑qubit rotations (phases)
            for i, phase in enumerate(layer.phases):
                qml.RZ(phase, wires=i)

            # squeezers → parameterized rotations with bounded values
            for i, (r, ph) in enumerate(zip(layer.squeeze_r, layer.squeeze_phi)):
                r_clipped = _clip(r, 5.0)
                qml.RX(r_clipped, wires=i)
                qml.RZ(ph, wires=i)

            # second beam splitter
            qml.CRX(theta, wires=0)
            qml.CRY(phi, wires=1)

            # displacements → additional rotations
            for i, (dr, dphi) in enumerate(zip(layer.displacement_r, layer.displacement_phi)):
                dr_clipped = _clip(dr, 5.0)
                qml.RY(dr_clipped, wires=i)
                qml.RZ(dphi, wires=i)

            # kerr → controlled‑phase
            for i, k in enumerate(layer.kerr):
                k_clipped = _clip(k, 1.0)
                qml.CZ(wires=(i, (i+1)%2))
                qml.RZ(k_clipped, wires=i)

        # Measurement: probability of |11⟩ as fraud indicator
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def __call__(self, **kwargs):
        return self.vqc(**kwargs)

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

__all__ = ["FraudLayerParameters", "FraudDetectionCircuit"]
