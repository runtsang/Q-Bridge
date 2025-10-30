import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_qml_program",
]

@dataclass
class FraudLayerParameters:
    """Container for a single photonic layer, reused for the QML circuit."""
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

def build_fraud_detection_qml_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device = qml.device("default.qubit", wires=2),
) -> qml.QNode:
    """Return a Pennylane QNode that emulates the photonic fraud‑detection circuit."""
    def circuit(x0: float, x1: float):
        # Encode input features into qubit rotations
        qml.RX(x0, wires=0)
        qml.RX(x1, wires=1)

        def _apply_layer(params: FraudLayerParameters, clip: bool = False):
            # Beam splitter analogue via RZ and CZ
            qml.RZ(params.bs_theta, wires=0)
            qml.CZ(wires=[0, 1])
            for i, phase in enumerate(params.phases):
                qml.RZ(phase, wires=i)
            for r, phi in zip(params.squeeze_r, params.squeeze_phi):
                qml.RZ(_clip(r, 5), wires=i)  # approximate squeezing
            for r, phi in zip(params.displacement_r, params.displacement_phi):
                qml.RZ(_clip(r, 5), wires=i)  # approximate displacement
            for k in params.kerr:
                qml.RZ(_clip(k, 1), wires=i)  # approximate Kerr

        # Input layer (unclipped)
        _apply_layer(input_params, clip=False)
        # Hidden layers (clipped)
        for layer in layers:
            _apply_layer(layer, clip=True)

        # Measurement
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return qml.QNode(circuit, dev)
