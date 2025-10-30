import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
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

def _apply_layer_qubit(wires: Sequence[int], params: FraudLayerParameters, clip: bool) -> None:
    theta, phi = params.bs_theta, params.bs_phi
    if clip:
        theta = _clip(theta, 5)
        phi = _clip(phi, 5)
    qml.RX(theta, wires=wires[0])
    qml.RX(phi, wires=wires[1])
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])
    for i, (r, phi_r) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(r if not clip else _clip(r, 5), wires=wires[i])
    qml.CNOT(wires[0], wires[1])
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])
    for i, (r, phi_r) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RZ(r if not clip else _clip(r, 5), wires=wires[i])
    for i, k in enumerate(params.kerr):
        qml.RZ(k if not clip else _clip(k, 1), wires=wires[i])

def build_fraud_detection_qnode(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: qml.Device | None = None,
) -> qml.QNode:
    if device is None:
        device = qml.device("default.qubit", wires=2)

    @qml.qnode(device)
    def fraud_qnode(x0: float, x1: float) -> np.ndarray:
        # Encode input features as rotations
        qml.RX(x0, wires=0)
        qml.RX(x1, wires=1)
        # First layer
        _apply_layer_qubit((0, 1), input_params, clip=False)
        # Subsequent layers
        for layer in layers:
            _apply_layer_qubit((0, 1), layer, clip=True)
        # Return expectation values of PauliZ on both wires
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    return fraud_qnode

class FraudDetectionEnhanced:
    """Quantum variational fraud detection circuit using PennyLane."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: qml.Device | None = None,
    ) -> None:
        self.qnode = build_fraud_detection_qnode(input_params, layers, device=device)

    def __call__(self, x: Sequence[float]) -> np.ndarray:
        """Return expectation values of PauliZ for the two wires."""
        return self.qnode(*x)

    def predict(self, x: Sequence[float]) -> float:
        """Combine the two expectation values into a single logit."""
        z_vals = self.__call__(x)
        return float((z_vals[0] + z_vals[1]) / 2.0)

__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
