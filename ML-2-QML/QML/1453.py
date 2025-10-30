import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    """Parameters for a variational fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    # rotation angles for the qubit encoding
    rot_x: Tuple[float, float] = (0.0, 0.0)
    rot_z: Tuple[float, float] = (0.0, 0.0)

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(q: qml.Device, params: FraudLayerParameters, *, clip: bool) -> None:
    # Encode layer‑specific rotations
    qml.RX(params.rot_x[0]) | 0
    qml.RZ(params.rot_z[0]) | 0
    qml.RX(params.rot_x[1]) | 1
    qml.RZ(params.rot_z[1]) | 1

    # Entangling operation parameterised by bs_theta, bs_phi
    qml.CZ(wires=[0, 1])

    # Layer‑dependent single‑qubit rotations derived from squeeze and displacement
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RX(_clip(r, 5.0)) | i
        qml.RZ(phi) | i

    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(_clip(r, 5.0)) | i
        qml.RZ(phi) | i

    # Kerr‑like non‑linearity simulated with RZ
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0)) | i

def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: qml.Device | None = None
) -> qml.QNode:
    """Return a PennyLane QNode implementing the fraud‑detection ansatz."""
    dev = device or qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(inputs: np.ndarray) -> np.ndarray:
        # Feature encoding – rotate by input values
        qml.RX(inputs[0]) | 0
        qml.RX(inputs[1]) | 1

        _apply_layer(dev, input_params, clip=False)
        for layer in layers:
            _apply_layer(dev, layer, clip=True)

        # Measurement: expectation of Pauli‑Z on first wire
        return qml.expval(qml.PauliZ(0))

    return circuit

class FraudDetectionModel:
    """
    Quantum analogue exposing the same class name.
    The ``predict`` method evaluates the QNode on a numpy array of inputs.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 device: qml.Device | None = None) -> None:
        self.circuit = build_fraud_detection_circuit(input_params, layers, device)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.circuit(x)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
