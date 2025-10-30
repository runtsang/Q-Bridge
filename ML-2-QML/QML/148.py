"""Quantum implementation of the fraud‑detection circuit using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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


class FraudDetectionModel:
    """Variational quantum circuit that mirrors the classical photonic architecture."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device=None,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.device = device or qml.device("default.qubit", wires=2)
        self._build_qnode()

    def _apply_layer(
        self, params: FraudLayerParameters, clip: bool, wires: Sequence[int]
    ) -> None:
        """Apply a photonic‑style layer using qubit gates."""
        # Beam‑splitter angles → RZ rotations
        theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
        phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
        qml.RZ(theta, wires=wires[0])
        qml.RZ(phi, wires=wires[1])

        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=wires[i])

        # Squeezing → RX & RY rotations
        for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_val = r if not clip else _clip(r, 5.0)
            ph_val = ph if not clip else _clip(ph, 5.0)
            qml.RX(r_val, wires=wires[i])
            qml.RY(ph_val, wires=wires[i])

        # Entanglement (approximate BSgate)
        qml.CNOT(wires=wires)

        # Repeat phases
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=wires[i])

        # Displacement → RZ rotations
        for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_val = r if not clip else _clip(r, 5.0)
            ph_val = ph if not clip else _clip(ph, 5.0)
            qml.RZ(r_val, wires=wires[i])

        # Kerr → RZ rotations
        for i, k in enumerate(params.kerr):
            k_val = k if not clip else _clip(k, 1.0)
            qml.RZ(k_val, wires=wires[i])

    def _build_qnode(self) -> None:
        @qml.qnode(self.device, interface="torch")
        def circuit(x1, x2, *layer_params):
            # Input encoding
            qml.RZ(x1, wires=0)
            qml.RZ(x2, wires=1)

            # First (unclipped) layer
            self._apply_layer(layer_params[0], clip=False, wires=[0, 1])

            # Subsequent layers (clipped)
            for idx, params in enumerate(layer_params[1:]):
                self._apply_layer(params, clip=True, wires=[0, 1])

            # Output: expectation of PauliZ on qubit 0
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of 2‑dimensional inputs."""
        outputs = []
        layer_params = [self.input_params] + self.layers
        for xi in x:
            out = self.qnode(
                xi[0].item(),
                xi[1].item(),
                *layer_params,
            )
            outputs.append(out)
        return torch.tensor(outputs).unsqueeze(-1)


__all__ = ["FraudDetectionModel"]
