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

class FraudDetectionEnhanced:
    """Quantum fraud detection model using Pennylane. Supports multi‑shot simulation
    and a tunable loss‑based training loop."""
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 shots: int = 1000) -> None:
        self.input_params = input_params
        self.layers_params = list(layers)
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=2, shots=self.shots)
        self.params = self._flatten_params()
        self.qnode = qml.QNode(lambda x, *flat_params: self._circuit(x, *flat_params),
                               self.dev,
                               interface="autograd")

    def _flatten_params(self) -> np.ndarray:
        arr = []
        for p in [self.input_params] + self.layers_params:
            arr.extend([p.bs_theta, p.bs_phi])
            arr.extend(p.phases)
            arr.extend(p.squeeze_r)
            arr.extend(p.squeeze_phi)
            arr.extend(p.displacement_r)
            arr.extend(p.displacement_phi)
            arr.extend(p.kerr)
        return np.array(arr, dtype=np.float64)

    def _circuit(self, x: np.ndarray, *flat_params: float) -> float:
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)

        idx = 0
        params_list = []
        for _ in [self.input_params] + self.layers_params:
            bs_theta = flat_params[idx]; idx += 1
            bs_phi = flat_params[idx]; idx += 1
            phases = (flat_params[idx], flat_params[idx + 1]); idx += 2
            squeeze_r = (flat_params[idx], flat_params[idx + 1]); idx += 2
            squeeze_phi = (flat_params[idx], flat_params[idx + 1]); idx += 2
            displacement_r = (flat_params[idx], flat_params[idx + 1]); idx += 2
            displacement_phi = (flat_params[idx], flat_params[idx + 1]); idx += 2
            kerr = (flat_params[idx], flat_params[idx + 1]); idx += 2
            params_list.append(FraudLayerParameters(
                bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
                displacement_r, displacement_phi, kerr))

        for params in params_list:
            qml.CNOT(wires=[0, 1])
            for j, phase in enumerate(params.phases):
                qml.PhaseShift(phase, wires=j)
            for j, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                qml.Squeezing(r, phi, wires=j)
            for j, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                qml.Displacement(r, phi, wires=j)
            for j, k in enumerate(params.kerr):
                qml.CZ(wires=[j, (j + 1) % 2])

        return qml.expval(qml.PauliZ(0))

    def loss(self, x: np.ndarray, y: float, *flat_params: float) -> float:
        pred = self.qnode(x, *flat_params)
        return (pred - y) ** 2

    def train(self, data_loader, epochs: int = 10, lr: float = 0.01) -> None:
        for epoch in range(epochs):
            for x, y in data_loader:
                grad_fn = qml.gradients.param_shift(self.loss)
                grads = grad_fn(x, y, *self.params)
                self.params = np.array(self.params) - lr * np.array(grads)

__all__ = ["FraudDetectionEnhanced", "FraudLayerParameters"]
