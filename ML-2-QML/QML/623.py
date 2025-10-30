import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable

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

class FraudDetection:
    """A PennyLane variational circuit that mirrors the photonic parameter set."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters], wires=2):
        self.device = qml.device("default.qubit", wires=wires)
        self.input_params = input_params
        self.layers = list(layers)
        self.qnode = qml.QNode(self._circuit, self.device)

    def _clip(self, value, bound):
        return max(-bound, min(bound, value))

    def _apply_layer(self, params: FraudLayerParameters, clip: bool):
        # Parametric rotations mimicking photonic operations
        qml.RZ(params.phases[0], wires=0)
        qml.RZ(params.phases[1], wires=1)
        qml.RY(params.squeeze_r[0], wires=0)
        qml.RY(params.squeeze_r[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params.displacement_r[0], wires=0)
        qml.RZ(params.displacement_r[1], wires=1)
        # Kerr‑like nonlinearity via a controlled‑Z (fixed for simplicity)
        for i, k in enumerate(params.kerr):
            angle = k if not clip else self._clip(k, 1.0)
            # The angle is ignored in this placeholder; a true implementation could use a parameterized CZ.
            qml.CZ(wires=[i, (i + 1) % 2])

    def _circuit(self):
        self._apply_layer(self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(layer, clip=True)
        return qml.expval(qml.PauliZ(0))

    def __call__(self):
        return self.qnode()
