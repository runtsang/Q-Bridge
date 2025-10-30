"""Quantum QCNN implementation using Pennylane's MLQNN."""

import pennylane as qml
import pennylane.numpy as np
from pennylane import qnn

def QCNN() -> qnn.MLQNN:
    """Return a hybrid quantum‑classical QCNN model."""
    n_qubits = 8
    dev = qml.device("default.qubit", wires=n_qubits)

    # Feature map: Z‑feature map with entangling CNOTs
    def feature_map(inputs):
        for i, x in enumerate(inputs):
            qml.RZ(x, wires=i)
        # Entangling layer to mix information
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])

    # Ansatz: 3 layers of RX/RY rotations followed by entangling CNOTs
    def ansatz(weights):
        idx = 0
        for _ in range(3):
            for w in range(n_qubits):
                qml.RX(weights[idx], wires=w); idx += 1
                qml.RY(weights[idx], wires=w); idx += 1
            # Entangling pattern
            for i in range(0, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        feature_map(inputs)
        ansatz(weights)
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (3 * n_qubits * 2,)}
    qnn_model = qnn.MLQNN(
        circuit=circuit,
        weight_shapes=weight_shapes,
        output_dim=1,
    )
    return qnn_model

__all__ = ["QCNN"]
