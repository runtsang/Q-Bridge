import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    phase_shift: float = 0.0

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(params: FraudLayerParameters, clip: bool = False) -> None:
    for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        angle = r if not clip else _clip(r, 5)
        qml.RY(angle, wires=i)
        qml.RZ(ph, wires=i)
    qml.CNOT(wires=[0, 1])
    for i, k in enumerate(params.kerr):
        angle = k if not clip else _clip(k, 1)
        qml.PhaseShift(angle, wires=i)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device | None = None,
) -> qml.QNode:
    if dev is None:
        dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs: np.ndarray) -> np.ndarray:
        qml.RY(inputs[0], wires=0)
        qml.RZ(inputs[1], wires=1)

        _apply_layer(input_params, clip=False)
        for layer in layers:
            _apply_layer(layer, clip=True)

        qml.PhaseShift(input_params.phase_shift, wires=0)
        qml.PhaseShift(input_params.phase_shift, wires=1)

        return qml.expval(qml.PauliZ(0))

    return circuit

__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
