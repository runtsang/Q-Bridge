"""Quantum QCNN implemented with PennyLane."""
from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
import numpy as onp

def conv_circuit(params: np.ndarray, wires: list[int]) -> None:
    """Parameterised 2‑qubit convolution block."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi / 2, wires=wires[0])

def pool_circuit(params: np.ndarray, wires: list[int]) -> None:
    """Parameterised 2‑qubit pooling block."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])

def _layer(circuit_fn, num_qubits: int, param_prefix: str):
    """Higher‑order factory that returns a layer applying a 2‑qubit block."""
    def layer(params: np.ndarray, wires: list[int]) -> None:
        idx = 0
        for q1, q2 in zip(wires[0::2], wires[1::2]):
            circuit_fn(params[idx : idx + 3], [q1, q2])
            idx += 3
        for q1, q2 in zip(wires[1::2], wires[2::2] + [wires[0]]):
            circuit_fn(params[idx : idx + 3], [q1, q2])
            idx += 3
    return layer

conv_layer = lambda num_qubits, prefix: _layer(conv_circuit, num_qubits, prefix)
pool_layer = lambda num_qubits, prefix: _layer(pool_circuit, num_qubits, prefix)

def QCNN(num_qubits: int = 8) -> qml.QNode:
    """Return a PennyLane QNode that implements the QCNN ansatz."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
        # Feature map
        qml.templates.feature_maps.ZFeatureMap(n_wires=num_qubits, reps=1)(inputs)

        # First convolution + pooling
        conv_layer(num_qubits, "c1")(weights["c1"], list(range(num_qubits)))
        pool_layer(num_qubits, "p1")(weights["p1"], list(range(num_qubits)))

        # Second convolution + pooling on the remaining qubits
        conv_layer(num_qubits // 2, "c2")(weights["c2"], list(range(num_qubits // 2, num_qubits)))
        pool_layer(num_qubits // 2, "p2")(weights["p2"], list(range(num_qubits // 2, num_qubits)))

        # Third convolution + pooling on the last two qubits
        conv_layer(num_qubits // 4, "c3")(weights["c3"], list(range(num_qubits // 4 * 3, num_qubits)))
        pool_layer(num_qubits // 4, "p3")(weights["p3"], list(range(num_qubits // 4 * 3, num_qubits)))

        return qml.expval(qml.PauliZ(0))

    return circuit

__all__ = ["QCNN"]
