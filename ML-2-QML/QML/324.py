"""Hybrid QCNN implemented with Pennylane and a classical post‑processing head."""
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Callable

# Number of qubits matches the classical input dimensionality
NUM_QUBITS = 8
OBSERVABLE = qml.PauliZ(0)  # single‑qubit measurement


def _conv_circuit(qubits: list[int], params: np.ndarray) -> Callable:
    """Two‑qubit convolution block parameterised by 3 angles."""
    def circuit():
        qml.RZ(-np.pi / 2, qubits[1])
        qml.CNOT(qubits[1], qubits[0])
        qml.RZ(params[0], qubits[0])
        qml.RY(params[1], qubits[1])
        qml.CNOT(qubits[0], qubits[1])
        qml.RY(params[2], qubits[1])
        qml.CNOT(qubits[1], qubits[0])
        qml.RZ(np.pi / 2, qubits[0])
    return circuit


def _pool_circuit(qubits: list[int], params: np.ndarray) -> Callable:
    """Two‑qubit pooling block parameterised by 3 angles."""
    def circuit():
        qml.RZ(-np.pi / 2, qubits[1])
        qml.CNOT(qubits[1], qubits[0])
        qml.RZ(params[0], qubits[0])
        qml.RY(params[1], qubits[1])
        qml.CNOT(qubits[0], qubits[1])
        qml.RY(params[2], qubits[1])
    return circuit


def _conv_layer(qubits: list[int], params: np.ndarray) -> Callable:
    """Apply convolution blocks pairwise across the qubit array."""
    def circuit():
        for i in range(0, len(qubits) - 1, 2):
            _conv_circuit(qubits[i:i + 2], params[i * 3 : (i + 1) * 3])()
    return circuit


def _pool_layer(qubits: list[int], params: np.ndarray) -> Callable:
    """Apply pooling blocks pairwise across the qubit array."""
    def circuit():
        for i in range(0, len(qubits) - 1, 2):
            _pool_circuit(qubits[i:i + 2], params[i * 3 : (i + 1) * 3])()
    return circuit


def _feature_map(x: np.ndarray) -> Callable:
    """Z‑feature map embedding classical data into the qubits."""
    def circuit():
        for i, val in enumerate(x):
            qml.RZ(val, i)
    return circuit


def QCNN() -> qml.QNode:
    """
    Construct a hybrid QCNN QNode.

    The circuit consists of:
      * A Z‑feature map for data encoding.
      * Three convolution‑pooling stages with independent weight
        parameters.
      * A single‑qubit measurement followed by a classical
        linear layer (implemented as a separate PyTorch module).
    """
    # Total trainable parameters: 3 layers × 8 qubits × 3 angles × 2 (conv & pool)
    num_params = 3 * NUM_QUBITS * 3 * 2

    dev = qml.device("default.qubit", wires=NUM_QUBITS)

    # Classical post‑processing head
    class ClassicalHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(NUM_QUBITS, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    head = ClassicalHead()

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Feature map
        _feature_map(inputs)()

        # Convolution‑pooling layers
        w_idx = 0
        for layer in range(3):
            conv_params = weights[w_idx : w_idx + NUM_QUBITS * 3]
            w_idx += NUM_QUBITS * 3
            _conv_layer(list(range(NUM_QUBITS)), conv_params)()

            pool_params = weights[w_idx : w_idx + NUM_QUBITS * 3]
            w_idx += NUM_QUBITS * 3
            _pool_layer(list(range(NUM_QUBITS)), pool_params)()

        # Measurement
        meas = qml.expval(OBSERVABLE)
        # Convert measurement into a vector for classical head
        meas_vec = qml.math.stack([meas for _ in range(NUM_QUBITS)])
        return head(meas_vec)

    # Initialise random weights
    init_weights = torch.randn(num_params, requires_grad=True)

    # Wrap the QNode to allow easy gradient‑based optimisation
    def hybrid_forward(inputs: torch.Tensor) -> torch.Tensor:
        return circuit(inputs, init_weights)

    return hybrid_forward
