"""Pennylane based variational fraud detection circuit."""

import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic (or variational) layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(
    wires: Sequence[int],
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    # Entangling operation approximating a beam‑splitter
    qml.CNOT(wires=[wires[0], wires[1]])
    # Phase rotations
    for i, phase in enumerate(params.phases):
        qml.RY(phase, wires=wires[i])
    # Squeezing (approximated by RZ)
    for i, (r, _) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(_clip(r, 5.0), wires=wires[i])
    # Displacement (approximated by RX)
    for i, (r, _) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(_clip(r, 5.0), wires=wires[i])
    # Kerr nonlinearity (approximated by Z rotation)
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0), wires=wires[i])

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    n_qubits: int = 2,
    device_name: str = "default.qubit",
) -> qml.QNode:
    """Create a Pennylane QNode for the hybrid fraud detection model."""
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x: np.ndarray) -> np.ndarray:
        # Encode classical features into rotation angles
        qml.templates.AngleEmbedding(x, wires=range(n_qubits))
        # First (unclipped) layer
        _apply_layer(range(n_qubits), input_params, clip=False)
        # Subsequent (clipped) layers
        for layer in layers:
            _apply_layer(range(n_qubits), layer, clip=True)
        # Measurement: expectation value of PauliZ on first qubit
        return qml.expval(qml.PauliZ(0))

    return circuit

class FraudDetectionModel:
    """Wraps a Pennylane variational circuit for fraud detection."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        n_qubits: int = 2,
        device_name: str = "default.qubit",
    ) -> None:
        self.circuit = build_fraud_detection_program(
            input_params, layers, n_qubits=n_qubits, device_name=device_name
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.circuit(x)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionModel",
]
