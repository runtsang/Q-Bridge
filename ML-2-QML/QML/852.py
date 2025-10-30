import pennylane as qml
import pennylane.numpy as np

def conv_circuit(params: np.ndarray, wires: list[int]) -> None:
    """Two‑qubit convolution unit used in each convolutional layer."""
    qml.RZ(-np.pi / 2, wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(params[0], wires[0])
    qml.RY(params[1], wires[1])
    qml.CNOT(wires[0], wires[1])
    qml.RY(params[2], wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(np.pi / 2, wires[0])

def pool_circuit(params: np.ndarray, wires: list[int]) -> None:
    """Two‑qubit pooling unit that discards one qubit via measurement‑inspired rotation."""
    qml.RZ(-np.pi / 2, wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(params[0], wires[0])
    qml.RY(params[1], wires[1])
    qml.CNOT(wires[0], wires[1])
    qml.RY(params[2], wires[1])

def qcnn(num_qubits: int = 8, dev: qml.Device | None = None):
    """Constructs a differentiable QCNN circuit as a PennyLane QNode."""
    if dev is None:
        dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: list[np.ndarray]) -> torch.Tensor:
        # Feature map: simple Z‑rotations per qubit
        for i in range(num_qubits):
            qml.RZ(inputs[i], wires=i)

        # Three convolution‑pooling stages
        for stage in range(3):
            # Convolution over adjacent qubit pairs
            for i in range(0, num_qubits, 2):
                conv_circuit(weights[stage][i // 2], wires=[i, i + 1])

            # Pooling: pairwise reduction over every second pair
            for i in range(0, num_qubits - 2, 4):
                pool_circuit(weights[stage][num_qubits // 2 + i // 4], wires=[i, i + 2])

        # Measurement on the first qubit
        return qml.expval(qml.PauliZ(0))

    return circuit

def QCNN():
    """
    Factory returning a fully‑parameterised QCNN QNode ready for hybrid training.
    The returned QNode expects (inputs, weights) where weights is a list of
    stage‑wise parameter arrays matching the convolution and pooling layers.
    """
    num_qubits = 8
    dev = qml.device("default.qubit", wires=num_qubits)

    # Initialise weights: 3 stages, each with (num_qubits/2) convolution params and
    # (num_qubits/4) pooling params, each param set of size 3.
    weights = [np.random.randn(num_qubits // 2 + num_qubits // 4, 3) for _ in range(3)]

    # Return the QNode together with the initial weight tensor for convenience
    return qcnn(num_qubits, dev), weights

__all__ = ["QCNN"]
