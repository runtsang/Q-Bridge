"""Fraud detection model – PennyLane variational implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, Any

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


@dataclass
class FraudLayerParameters:
    """Same parameter layout as the classical version, now interpreted as variational angles."""
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


def _apply_layer(
    params: FraudLayerParameters,
    clip: bool,
    qdev: qml.Device,
    wires: Sequence[int],
) -> None:
    """Translate photonic parameters into qubit gates."""
    # Entangle via a controlled‑Z approximating a beam‑splitter
    qml.CZ(wires=[wires[0], wires[1]])
    # Phase shifts
    for w, phi in zip(wires, params.phases):
        qml.RZ(phi, wires=w)

    # “Squeezing” → two‑qubit rotation (approximation)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_eff = _clip(r, 5.0) if clip else r
        qml.RX(r_eff, wires=wires[i])
        qml.RZ(phi, wires=wires[i])

    # Additional entanglement
    qml.CZ(wires=[wires[0], wires[1]])

    # Phase shifts again
    for w, phi in zip(wires, params.phases):
        qml.RZ(phi, wires=w)

    # Displacement → single‑qubit rotation
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_eff = _clip(r, 5.0) if clip else r
        qml.RX(r_eff, wires=wires[i])
        qml.RZ(phi, wires=wires[i])

    # Kerr → ZZ rotation
    for i, k in enumerate(params.kerr):
        k_eff = _clip(k, 1.0) if clip else k
        qml.RZ(k_eff, wires=wires[i])


class FraudDetection:
    """
    PennyLane implementation of the fraud‑detection model.
    Constructs a variational circuit mirroring the classical layer layout.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        device: Optional[qml.Device] = None,
        qnode: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.device = device or qml.device("default.qubit", wires=2)
        self.qnode = qnode or self._build_qnode()

    def _build_qnode(self) -> Callable[..., Any]:
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: np.ndarray) -> np.ndarray:
            # encode inputs into qubit states (simple amplitude encoding)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.RY(np.arcsin(inputs[0]), wires=0)
            qml.RY(np.arcsin(inputs[1]), wires=1)

            # input layer (no clipping)
            _apply_layer(self.input_params, clip=False, qdev=self.device, wires=[0, 1])

            # hidden layers (clipped)
            for layer in self.layers:
                _apply_layer(layer, clip=True, qdev=self.device, wires=[0, 1])

            # final measurement
            return qml.expval(qml.PauliZ(0))
        return circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the variational circuit."""
        return torch.sigmoid(self.qnode(inputs))

    def loss(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Binary cross‑entropy loss."""
        logits = torch.tensor(logits, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        return F.binary_cross_entropy(logits, targets)

__all__ = ["FraudDetection", "FraudLayerParameters"]
