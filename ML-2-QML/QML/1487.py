"""Quantum estimator based on a variational circuit with entanglement and expectation read‑out.

The implementation uses Pennylane and supports arbitrary qubit counts,
parameterized Ry rotations, and a configurable entanglement pattern.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp
from pennylane.measurements import ExpectationSample

def EstimatorQNN(
    num_qubits: int = 2,
    entanglement: str = "circular",
    depth: int = 2,
    observable: str = "PauliZ",
) -> qml.QNode:
    """
    Construct a variational quantum circuit as a QNode.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str
        Entanglement pattern ('circular', 'full', 'none').
    depth : int
        Number of variational layers.
    observable : str
        Pauli string for expectation measurement (e.g., 'PauliX', 'PauliY', 'PauliZ', 'PauliXYZ').
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Data encoding: apply Ry to each qubit
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for d in range(depth):
            for i in range(num_qubits):
                qml.RY(weights[d, i], wires=i)
            # Entangling layer
            if entanglement == "full":
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qml.CNOT(wires=[i, j])
            elif entanglement == "circular":
                for i in range(num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % num_qubits])

        return qml.expval(getattr(qml, observable)(wires=range(num_qubits)))

    return circuit


__all__ = ["EstimatorQNN"]
