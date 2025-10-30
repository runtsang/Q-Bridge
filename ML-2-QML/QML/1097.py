"""Pennylane variational circuit that mirrors the photonic fraud‑detection layers.

The circuit maps each FraudLayerParameters instance to a sequence of
parameterised single‑qubit rotations and a two‑qubit beam‑splitter
equivalent (implemented as a controlled‑phase gate).  The module
provides a ``FraudDetectionQuantumCircuit`` class that can be used
directly in a hybrid training loop with PyTorch or any other
autograd‑enabled framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
from pennylane import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (kept for API compatibility)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _apply_layer(
    params: FraudLayerParameters,
    *,
    clip: bool,
    wires: Sequence[int],
    dev: qml.Device,
) -> None:
    """Translate a photonic layer into Pennylane gates."""
    # Beam‑splitter equivalent: a controlled‑phase gate with angle 2*theta
    theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
    phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
    qml.CZ(wires=wires, control_wires=[wires[0]], control_values=[0])  # placeholder

    # Single‑qubit rotations (phases)
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])

    # Squeezing → rotation+phase (approximation)
    for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_eff = r if not clip else _clip(r, 5.0)
        qml.RX(r_eff, wires=wires[i])
        qml.RZ(ph, wires=wires[i])

    # Displacement → rotation
    for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_eff = r if not clip else _clip(r, 5.0)
        qml.RY(r_eff, wires=wires[i])
        qml.RZ(ph, wires=wires[i])

    # Kerr → ZZ rotation (approximation)
    for i, k in enumerate(params.kerr):
        k_eff = k if not clip else _clip(k, 1.0)
        qml.RZ(k_eff, wires=wires[i])


class FraudDetectionQuantumCircuit:
    """Variational circuit that emulates the photonic fraud‑detection model."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dev: qml.Device | None = None,
    ) -> None:
        self.dev = dev or qml.device("default.qubit", wires=2)
        self.input_params = input_params
        self.layers = list(layers)

    def circuit(self) -> qml.QNode:
        """Return a Pennylane QNode that can be differentiated by autograd."""
        @qml.qnode(self.dev, interface="torch")
        def _circ(x: np.ndarray) -> np.ndarray:
            # Encode classical input as rotations
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)

            # Apply first layer (unclipped)
            _apply_layer(self.input_params, clip=False, wires=[0, 1], dev=self.dev)

            # Remaining layers (clipped)
            for layer in self.layers:
                _apply_layer(layer, clip=True, wires=[0, 1], dev=self.dev)

            # Measurement: expectation of PauliZ on both wires
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        return _circ

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the circuit on a single 2‑dimensional input."""
        return self.circuit()(x)

__all__ = ["FraudLayerParameters", "FraudDetectionQuantumCircuit"]
