"""Quantum-enhanced QCNN using PennyLane."""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

# 8‑qubit device for simulation
dev = qml.device("default.qubit", wires=8)


def conv_circuit(params, wires):
    """Two‑qubit convolution block parameterized by three angles."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi / 2, wires=wires[0])


def conv_layer(params, wires):
    """Apply convolution blocks across alternating qubit pairs."""
    # First pass over even‑odd pairs
    for i in range(0, len(wires) - 1, 2):
        conv_circuit(params[i : i + 3], wires[i : i + 2])
    # Second pass over odd‑even pairs
    for i in range(1, len(wires) - 1, 2):
        conv_circuit(params[i : i + 3], wires[i : i + 2])


def pool_circuit(params, wires):
    """Two‑qubit pooling block that discards one qubit by ignoring it after the circuit."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])
    # The second qubit is effectively discarded for the next stage


def pool_layer(params, sources, sinks):
    """Apply pooling blocks to specified source‑sink pairs."""
    for src, sink in zip(sources, sinks):
        idx = sources.index(src)
        pool_circuit(params[idx * 3 : (idx + 1) * 3], wires=[src, sink])


def feature_map(x):
    """Z feature map applied to all qubits."""
    for i, xi in enumerate(x):
        qml.RZ(2 * xi, wires=i)


@qml.qnode(dev, interface="torch")
def qcnn_node(params, x):
    """Hybrid QCNN node returning the expectation value of Pauli‑Z on qubit 0."""
    feature_map(x)

    # Convolution–pooling stages
    conv_layer(params["c1"], dev.wires)
    pool_layer(params["p1"], [0, 1, 2, 3], [4, 5, 6, 7])

    conv_layer(params["c2"], dev.wires[4:8])
    pool_layer(params["p2"], [0, 1], [2, 3])

    conv_layer(params["c3"], dev.wires[6:8])
    pool_layer(params["p3"], [0], [1])

    return qml.expval(qml.PauliZ(0))


def QCNNQuantum() -> dict:
    """Return a parameter dictionary and the QNode for training."""
    # Random initialization of trainable parameters
    params = {
        "c1": np.random.randn(8 * 3),
        "p1": np.random.randn(4 * 3),
        "c2": np.random.randn(4 * 3),
        "p2": np.random.randn(2 * 3),
        "c3": np.random.randn(2 * 3),
        "p3": np.random.randn(1 * 3),
    }
    return {"params": params, "qnode": qcnn_node}


__all__ = ["QCNNQuantum", "qcnn_node"]
