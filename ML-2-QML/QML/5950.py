import pennylane as qml
import numpy as np
from.FraudDetection import FraudLayerParameters

class FraudDetectionModel:
    """
    Quantum‑classical fraud detection model that mirrors the photonic layer
    definitions of the seed but uses a qubit variational circuit.
    """

    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: list[FraudLayerParameters],
                 dev_name: str = "default.qubit",
                 wires: int = 2):
        self.wires = wires
        self.dev = qml.device(dev_name, wires=wires, shots=1024)
        self.input_params = input_params
        self.layers = layers
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, x: np.ndarray, params: list[FraudLayerParameters]) -> float:
        # Input encoding
        for i in range(self.wires):
            qml.RX(x[i], wires=i)

        # Apply layers
        for layer in params:
            self._apply_layer(layer)

        # Measurement
        return qml.expval(qml.PauliZ(0))

    def _apply_layer(self, layer: FraudLayerParameters):
        """
        Apply a single photonic‑style layer using qubit gates.
        """
        # Beam splitter analogue: CNOT entanglement
        qml.CNOT(wires=[0, 1])

        # Phase shifts -> RZ
        for i, phase in enumerate(layer.phases):
            qml.RZ(phase, wires=i)

        # Squeezing -> RX with amplitude
        for i, (r, phi) in enumerate(zip(layer.squeeze_r, layer.squeeze_phi)):
            qml.RX(r * np.cos(phi), wires=i)

        # Displacement -> RY
        for i, (r, phi) in enumerate(zip(layer.displacement_r, layer.displacement_phi)):
            qml.RY(r * np.sin(phi), wires=i)

        # Kerr -> RZ
        for i, k in enumerate(layer.kerr):
            qml.RZ(k, wires=i)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Return the expectation value for a batch of inputs.
        """
        params = [self.input_params] + self.layers
        return np.array([self.qnode(xi, params) for xi in x])

    def compute_hybrid_loss(self, logits: np.ndarray, quantum_out: np.ndarray, targets: np.ndarray) -> float:
        """
        Hybrid loss that penalises both the classical predictions (logits)
        and the quantum circuit expectation values.  The loss is a weighted
        sum of binary cross‑entropy and mean‑squared error.
        """
        eps = 1e-8
        probs = 1 / (1 + np.exp(-logits))
        bce = -np.mean(targets * np.log(probs + eps) + (1 - targets) * np.log(1 - probs + eps))
        mse = np.mean((quantum_out - logits) ** 2)
        return bce + 0.5 * mse
