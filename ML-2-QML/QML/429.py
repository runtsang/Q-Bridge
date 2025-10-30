"""FraudDetectionNetwork: quantum‑classical hybrid model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import torch
import torch.nn.functional as F


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


class FraudDetectionNetwork:
    """Hybrid quantum‑classical fraud‑detection model.

    The quantum part is a PennyLane variational circuit that mirrors the
    photonic primitives: beam‑splitters → RX/RZ, squeezers → RY,
    displacements → RX, and Kerr gates → RZ with a small phase.  A
    classical linear layer maps the two expectation values to a single score.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str = "default.qubit",
        wires: int = 2,
        *,
        seed: int | None = None,
    ) -> None:
        self.device = qml.device(device, wires=wires, shots=None)
        self.input_params = input_params
        self.layers = list(layers)
        self._build_circuit()
        self.readout = torch.nn.Linear(2, 1, bias=True)

    def _apply_layer(
        self,
        params: FraudLayerParameters,
        clip: bool = True,
    ) -> None:
        """Append a photonic‑style layer to the circuit."""
        theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
        phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
        # Beam‑splitter analogue
        qml.RX(theta, wires=0)
        qml.RX(phi, wires=1)

        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r = r if not clip else _clip(r, 5.0)
            qml.RY(r, wires=i)

        # Repeat beam‑splitter
        qml.RX(theta, wires=0)
        qml.RX(phi, wires=1)

        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        for i, (r, _) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r = r if not clip else _clip(r, 5.0)
            qml.RX(r, wires=i)

        for i, k in enumerate(params.kerr):
            k = k if not clip else _clip(k, 1.0)
            qml.RZ(k, wires=i)

    def _build_circuit(self) -> None:
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            """Quantum circuit producing expectation values of Z on each qubit."""
            # Encode a 2‑dimensional feature vector as rotations
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)

            # First layer (unclipped)
            self._apply_layer(self.input_params, clip=False)

            # Subsequent layers (clipped)
            for layer in self.layers:
                self._apply_layer(layer, clip=True)

            return [qml.expval(qml.PauliZ(i)) for i in range(self.device.wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantum measurement followed by a linear readout."""
        if x.dim()!= 1 or x.size(0)!= 2:
            raise ValueError("Input must be a 1‑D tensor of shape (2,).")
        q_out = self.circuit(x)
        return self.readout(q_out)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross‑entropy loss; optional weight‑decay can be added externally."""
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)
