"""Variational quantum fraud‑detection model based on PennyLane.

The quantum part implements a two‑qubit circuit whose gates are
parameterised by the same `FraudLayerParameters` used in the
classical seed.  The circuit encodes the input feature vector via
RY rotations, applies a stack of parameterised layers, and
measures the expectation value of Pauli‑Z on both qubits.  A
small classical head converts the two expectation values into a
binary fraud‑score.

The class is fully Torch‑compatible and can be used with
PyTorch optimisers or PennyLane optimisers.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

# --------------------------------------------------------------------------- #
#  Quantum‑parameter mapping
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters defining a single layer in the quantum circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [−bound, bound]."""
    return max(-bound, min(bound, value))

class FraudDetectionModel(nn.Module):
    """Quantum‑classical hybrid fraud‑detection model."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        *,
        device: str = "default.qubit",
        wires: int = 2,
        head_hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_params = input_params
        self.layer_params = list(layer_params)
        self.device = device
        self.wires = wires

        # Classical head that maps the two expectation values to a score
        self.head = nn.Sequential(
            nn.Linear(2, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

        # Build the QNode
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _apply_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        """Append parameterised gates to the circuit."""
        # Beam splitter analogue – a rotation on each qubit
        qml.RZ(params.bs_theta, wires=0)
        qml.RZ(params.bs_phi, wires=1)

        # Entangling operation (CNOT as a stand‑in for BS)
        qml.CNOT(wires=[0, 1])

        # Squeezing analogue – rotation around Y
        qml.RY(_clip(params.squeeze_r[0], 5.0), wires=0)
        qml.RY(_clip(params.squeeze_r[1], 5.0), wires=1)

        # Phase shifters
        qml.RZ(_clip(params.phases[0], 5.0), wires=0)
        qml.RZ(_clip(params.phases[1], 5.0), wires=1)

        # Displacement analogue – additional rotation around Z
        qml.RZ(_clip(params.displacement_r[0], 5.0), wires=0)
        qml.RZ(_clip(params.displacement_r[1], 5.0), wires=1)

        # Kerr non‑linearity analogue – small Z rotations
        qml.RZ(_clip(params.kerr[0], 1.0), wires=0)
        qml.RZ(_clip(params.kerr[1], 1.0), wires=1)

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum circuit that encodes `x` and returns a 2‑dimensional
        vector of expectation values.
        """
        # Encode classical input via RY rotations
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Input layer
        self._apply_layer(self.input_params, clip=False)

        # Hidden layers
        for params in self.layer_params:
            self._apply_layer(params, clip=True)

        # Measurements
        exp1 = qml.expval(qml.PauliZ(0))
        exp2 = qml.expval(qml.PauliZ(1))
        return torch.stack([exp1, exp2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the quantum circuit and apply the classical head."""
        q_output = self.qnode(x)
        return self.head(q_output)

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionModel",
]
