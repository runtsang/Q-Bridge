import pennylane as qml
import pennylane.numpy as np
from pennylane.optimize import Adam
from pennylane import qnode
from pennylane.measurement import StateFn


def _conv_layer(layer_idx: int, params: np.ndarray, wires: list[int]) -> None:
    """Apply a 2‑qubit convolution unitary on each adjacent pair of wires."""
    for i in range(0, len(wires) - 1, 2):
        w1, w2 = wires[i], wires[i + 1]
        # parameters for this pair
        a, b, c = params[layer_idx, i // 2, :]
        qml.RZ(-np.pi / 2, wires=w2)
        qml.CNOT(wires=[w2, w1])
        qml.RZ(a, wires=w1)
        qml.RY(b, wires=w2)
        qml.CNOT(wires=[w1, w2])
        qml.RY(c, wires=w2)
        qml.CNOT(wires=[w2, w1])
        qml.RZ(np.pi / 2, wires=w1)


def _pool_layer(layer_idx: int, params: np.ndarray, wires: list[int]) -> None:
    """Apply a 2‑qubit pooling operation that collapses one qubit."""
    for i in range(0, len(wires) - 1, 2):
        w1, w2 = wires[i], wires[i + 1]
        a, b, c = params[layer_idx, i // 2, :]
        qml.RZ(-np.pi / 2, wires=w2)
        qml.CNOT(wires=[w2, w1])
        qml.RZ(a, wires=w1)
        qml.RY(b, wires=w2)
        qml.CNOT(wires=[w1, w2])
        qml.RY(c, wires=w2)
        # After this we keep only the first qubit as the pooled result
        # The second qubit will be discarded in the next layer


def _feature_map(x: np.ndarray, wires: list[int]) -> None:
    """Encode data into the quantum state."""
    for i, val in enumerate(x):
        qml.RZ(val, wires=wires[i])
    # Entangle all wires strongly
    for i in range(0, len(wires) - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def QCNN(num_qubits: int = 8, depth: int = 3, device: str = "default.qubit") -> qml.QNode:
    """
    Build a hybrid variational circuit that mirrors the classical QCNN.

    Args:
        num_qubits: Number of qubits used for the feature map and ansatz.
        depth: Number of convolution + pooling stages.
        device: Pennylane device string.

    Returns:
        A QNode that outputs a single probability in [0, 1].
    """
    dev = qml.device(device, wires=num_qubits)

    # Parameters: one 3‑tuple per 2‑qubit block per layer
    param_shape = (depth, num_qubits // 2, 3)
    init_params = np.random.uniform(0, 2 * np.pi, size=param_shape)

    @qml.qnode(dev, interface="autograd")
    def circuit(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        _feature_map(x, wires=range(num_qubits))

        # Convolution and pooling stages
        for d in range(depth):
            _conv_layer(d, params, wires=range(num_qubits))
            # Pooling reduces the effective qubit count by half
            # We simply ignore the second qubit of each pair afterwards
            # by not using it in the next layer.
            # The circuit continues to use all qubits but the measurement
            # will only be taken from the first qubit of each pair.
            # This mimics a “classical” pooling step.
            # No explicit discarding is required in Pennylane.
            # The observable is chosen accordingly.
            # Pooling is effectively a measurement of the second qubit
            # and a reset of the first qubit; we approximate this by
            # measuring the first qubit after the conv block.
            # The observable is a Pauli‑Z on the first qubit.
            # The output of the whole circuit is the expectation value
            # of this observable, which is then passed through a sigmoid.
            # Note: In a real implementation one might use a classical
            # post‑processing step to combine pooled results.
            # For brevity we keep a single measurement per layer.
            if d < depth - 1:
                # Prepare for next layer: keep only the first qubit of each pair
                # by resetting the second qubit to |0> (approximate).
                for i in range(1, num_qubits, 2):
                    qml.Reset(wires=i)

        # Final measurement on the first qubit
        return qml.expval(qml.PauliZ(0))

    def model(x: np.ndarray) -> np.ndarray:
        """Convenience wrapper that returns a probability."""
        raw = circuit(x, init_params)
        # Map expectation value from [-1, 1] to [0, 1]
        return 0.5 * (raw + 1)

    return model
