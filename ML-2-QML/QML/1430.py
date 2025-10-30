import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Callable

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
    dropout_rate: float = 0.0  # retained for API parity

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(wires: List[int], params: FraudLayerParameters, clip: bool) -> None:
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = r if not clip else _clip(r, 5)
        qml.Sgate(r_val, phi, wires=wires[i])
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_val = r if not clip else _clip(r, 5)
        qml.Dgate(r_val, phi, wires=wires[i])
    for i, k in enumerate(params.kerr):
        k_val = k if not clip else _clip(k, 1)
        qml.Kgate(k_val, wires=wires[i])

def build_fraud_detection_qnode(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> Callable[[np.ndarray], np.ndarray]:
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(inputs: np.ndarray) -> np.ndarray:
        qml.StatePrep(inputs, wires=[0, 1])
        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)
        return qml.expval(qml.PauliZ(0))

    return circuit

class FraudDetectionQuantumModel:
    """
    Quantum‑enhanced fraud‑detection model built on a Pennylane variational circuit.
    The circuit reproduces the photonic layer structure while returning a single‑bit
    expectation value that can be interpreted as a fraud score.
    """
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        self.circuit = build_fraud_detection_qnode(input_params, layers)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.circuit(inputs)

    def loss(self, inputs: np.ndarray, labels: np.ndarray) -> float:
        preds = self.circuit(inputs)
        return np.mean((preds - labels) ** 2)

__all__ = ["FraudLayerParameters", "FraudDetectionQuantumModel", "build_fraud_detection_qnode"]
