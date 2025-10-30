import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane_qiskit import QiskitDevice

class QCNNHybrid:
    """
    A variational QCNN with three convolution–pooling stages and a
    noise‑augmented measurement layer.  The circuit is built on 8 qubits,
    each convolution block uses a parameterised two‑qubit unitary,
    followed by a pooling block that measures a pair of qubits and
    discards them, mimicking the feature map reduction in the
    classical analogue.
    """
    def __init__(self, shots: int = 8192, noise: bool = True) -> None:
        self.dev = QiskitDevice(backend="qasm_simulator",
                                shots=shots,
                                noise=noise)
        self.num_qubits = 8
        self.circuit = self._build_circuit()

    def _conv_block(self, idx: int, params: np.ndarray) -> None:
        # Two‑qubit parameterised unitary on qubits (2i, 2i+1)
        qml.RZ(params[0], wires=2*idx)
        qml.CNOT(wires=[2*idx, 2*idx+1])
        qml.RZ(params[1], wires=2*idx+1)
        qml.CNOT(wires=[2*idx+1, 2*idx])
        qml.RZ(params[2], wires=2*idx)

    def _pool_block(self, idx: int, params: np.ndarray) -> None:
        # Measure first qubit and discard it; second qubit remains
        qml.RZ(params[0], wires=2*idx)
        qml.CNOT(wires=[2*idx, 2*idx+1])
        qml.RZ(params[1], wires=2*idx+1)
        qml.measure(wires=2*idx)

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(inputs: np.ndarray, weights: np.ndarray):
            # Feature map
            for i, x in enumerate(inputs):
                qml.RY(x, wires=i)
            # Convolution‑pooling stages
            w_idx = 0
            # Stage 1: 4 conv + pool
            for i in range(4):
                self._conv_block(i, weights[w_idx:w_idx+3])
                w_idx += 3
            for i in range(4):
                self._pool_block(i, weights[w_idx:w_idx+2])
                w_idx += 2
            # Stage 2: 2 conv + pool
            for i in range(2):
                self._conv_block(i, weights[w_idx:w_idx+3])
                w_idx += 3
            for i in range(2):
                self._pool_block(i, weights[w_idx:w_idx+2])
                w_idx += 2
            # Stage 3: 1 conv + pool
            self._conv_block(0, weights[w_idx:w_idx+3])
            w_idx += 3
            self._pool_block(0, weights[w_idx:w_idx+2])
            # Measurement
            return qml.expval(qml.PauliZ(0))
        return circuit

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the variational QCNN.  Returns the expectation
        value of Pauli‑Z on qubit 0 as the output probability.
        """
        num_params = self.circuit.num_params
        weights = pnp.random.uniform(0, 2*np.pi, num_params)
        return self.circuit(inputs, weights).reshape(-1, 1)

def QCNNHybrid() -> QCNNHybrid:
    """
    Factory returning a fully configured QCNNHybrid QNode.
    """
    return QCNNHybrid()

__all__ = ["QCNNHybrid"]
