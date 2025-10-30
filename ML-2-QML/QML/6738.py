import pennylane as qml
import numpy as np
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class FraudDetectionHybrid:
    """
    Quantum‑enhanced fraud detection model.
    Implements a variational circuit that mirrors the classical
    two‑mode structure. The circuit is wrapped in a PennyLane
    device and returns a binary classification via a single
    expectation value.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 n_layers: int = 2):
        self.n_layers = n_layers
        self.input_params = input_params
        self.layers = list(layers)
        self.device = qml.device("default.qubit", wires=2)
        self.qnode = qml.QNode(self._circuit, self.device)

    def _circuit(self, params: np.ndarray):
        # Encode input parameters into state
        qml.BasisState([0, 0], wires=[0, 1])
        # Apply input layer
        self._apply_layer(0, self.input_params, clip=False)
        # Variational layers
        for i in range(self.n_layers):
            qml.StronglyEntanglingLayers(params[i], wires=[0, 1])
        # Apply subsequent layers
        for layer in self.layers:
            self._apply_layer(0, layer, clip=True)
        return qml.expval(qml.PauliZ(0))

    def _apply_layer(self, wire: int, params: FraudLayerParameters, clip: bool):
        # Simplified mapping: use rotation gates for phases
        for i, phase in enumerate(params.phases):
            qml.RX(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.RY(r if not clip else _clip(r, 5), wires=i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.RZ(r if not clip else _clip(r, 5), wires=i)
        for i, k in enumerate(params.kerr):
            qml.RX(k if not clip else _clip(k, 1), wires=i)

    def __call__(self, params: np.ndarray) -> float:
        return self.qnode(params)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
