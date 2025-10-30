import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp
from typing import Callable

def QCNN() -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Builds a deep, parameter‑shift compatible QCNN ansatz using Pennylane.
    The circuit is parameterized by a weight vector and accepts classical
    feature vectors via AngleEmbedding. It returns the expectation value
    of PauliZ on wire 0, suitable for binary classification.

    The ansatz consists of:
        * 3 convolutional layers (2‑qubit rotations + CNOTs)
        * 3 pooling layers (2‑qubit rotations + CNOTs)
    Each layer uses a fresh set of parameters; the total count is
    3 * (8*2 + 4*2) = 112 real parameters.

    The function is JAX‑backed for efficient gradient evaluation via
    the parameter‑shift rule, enabling integration with classical optimizers.
    """
    dev = qml.device("default.qubit", wires=8)

    def conv_layer(wires, params):
        """Two‑qubit convolution block."""
        for i in range(0, len(wires), 2):
            qml.RZ(params[i], wires=wires[i])
            qml.RY(params[i + 1], wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    def pool_layer(wires, params):
        """Two‑qubit pooling block."""
        for i in range(0, len(wires), 2):
            qml.RZ(params[i], wires=wires[i])
            qml.RY(params[i + 1], wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    @qml.qnode(dev, interface="jax")
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
        # Feature encoding
        qml.templates.AngleEmbedding(inputs, wires=range(8))

        # Count of parameters per layer
        conv_params = 8 * 2      # 2 rotations per qubit
        pool_params = 4 * 2

        idx = 0
        for _ in range(3):  # 3 conv‑pool cycles
            conv_layer(range(8), weights[idx : idx + conv_params])
            idx += conv_params
            pool_layer(range(8), weights[idx : idx + pool_params])
            idx += pool_params

        return qml.expval(qml.PauliZ(0))

    return circuit

__all__ = ["QCNN"]
