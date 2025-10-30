"""Quantum hybrid fraud detection model using PennyLane and StrawberryFields.

The circuit implements the same layer structure as the seed but replaces the
classical linear transformations with a parameterised photonic circuit.
A small classical linear head maps the two‑mode output to a single score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.strawberryfields as sf
import torch
from torch import nn

# Device: 2‑mode Fock space with cutoff 10
dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=10)

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
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5), phi, wires=wires[i])
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5), phi, wires=wires[i])
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1), wires=wires[i])

class FraudDetectionModel(nn.Module):
    """Variational photonic fraud detection network.

    The network consists of a variational circuit built from a stack of
    photonic layers followed by a classical linear head.  The circuit is
    implemented as a PennyLane QNode so that gradients can be propagated
    through the quantum part for end‑to‑end training.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.input_params = input_params
        self.layers = list(layers)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Encode the two‑dimensional classical input as displacements
            qml.Displacement(inputs[0], 0.0, wires=0)
            qml.Displacement(inputs[1], 0.0, wires=1)
            # First photonic layer (unclipped)
            _apply_layer([0, 1], self.input_params, clip=False)
            # Subsequent layers
            for layer in self.layers:
                _apply_layer([0, 1], layer, clip=True)
            # Measure photon number in each mode
            return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

        self.circuit = circuit
        # Classical head
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Run the quantum circuit
        q_out = self.circuit(x)
        # Pass through classical head
        return self.head(q_out)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
