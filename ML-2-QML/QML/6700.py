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

class FraudDetection__gen557:
    """Quantum variational fraud detection model."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layer_params: Iterable[FraudLayerParameters],
                 num_qubits: int = 2,
                 num_layers: int = 2,
                 device: str = 'default.qubit'):
        self.device = qml.device(device, wires=num_qubits)
        self.input_params = input_params
        self.layer_params = list(layer_params)
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.params = self._initialize_params()
        self.qnode = self._build_qnode()

    def _initialize_params(self) -> np.ndarray:
        """Flatten all layer parameters into a single variational array."""
        init = []
        for p in self.layer_params:
            init.extend([p.bs_theta, p.bs_phi] + list(p.phases) +
                        list(p.squeeze_r) + list(p.squeeze_phi) +
                        list(p.displacement_r) + list(p.displacement_phi) +
                        list(p.kerr))
        return np.array(init, requires_grad=True)

    def _feature_map(self, x: np.ndarray):
        """Encode 2‑dimensional input into rotations."""
        qml.RY(x[0], wires=0)
        qml.RZ(x[1], wires=1)

    def _variational_layer(self, params, layer_idx: int):
        """Apply a layer of rotations followed by entanglement."""
        offset = layer_idx * 14  # 14 parameters per layer
        for i in range(self.num_qubits):
            qml.RZ(params[offset + 2 * i], wires=i)
            qml.RY(params[offset + 2 * i + 1], wires=i)
        qml.CNOT(wires=[0, 1])

    def _build_qnode(self):
        @qml.qnode(self.device, interface='autograd')
        def circuit(x, params):
            self._feature_map(x)
            for i in range(self.num_layers):
                self._variational_layer(params, i)
            return qml.expval(qml.PauliZ(0))
        return circuit

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.qnode(x, self.params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.forward(x)
        return (probs > 0).astype(np.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layers={self.num_layers}, qubits={self.num_qubits})"

__all__ = ["FraudLayerParameters", "FraudDetection__gen557"]
