import pennylane as qml
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

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

class FraudDetectionAdv:
    '''Quantum fraud detection model with variational circuit and adaptive measurement.'''

    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 dev_name: str = "default.qubit",
                 shots: int = 1000):
        self.dev = qml.device(dev_name, wires=2, shots=shots)
        self.input_params = input_params
        self.layers = list(layers)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x: Tuple[float, float], var_params: Sequence[float]):
        # Encode classical input via rotation gates
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        # Variational rotations
        for w, theta in enumerate(var_params):
            qml.RX(theta, wires=w)
        # Apply photonic layers
        for params in [self.input_params] + self.layers:
            self._apply_layer(params)
        # Adaptive measurement: expectation of PauliZ on wire 0
        return qml.expval(qml.PauliZ(0))

    def _apply_layer(self, params: FraudLayerParameters):
        # Use a CNOT as a placeholder for BS gate
        qml.CNOT(wires=[0,1])
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)
        for theta in params.kerr:
            qml.RX(theta, wires=0)

    def predict(self, x: Tuple[float, float], var_params: Sequence[float]):
        return self.qnode(x, var_params)

    def get_parameters(self):
        return [params for params in self.input_params.__dict__.values()] + \
               [p for layer in self.layers for p in layer.__dict__.values()]

    def set_parameters(self, param_list):
        # Not implemented: placeholder for future extension
        pass

__all__ = ["FraudDetectionAdv", "FraudLayerParameters"]
