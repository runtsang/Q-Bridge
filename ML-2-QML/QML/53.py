"""Hybrid quantum estimator using PennyLane with entangled variational circuit."""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import List

def EstimatorQNN(num_qubits: int = 4, entanglement: str = "circular") -> qml.QNode:
    """
    Return a PennyLane QNode that implements a variational quantum circuit
    for regression. The circuit is entangled across all qubits and uses
    trainable rotation angles and a multi‑qubit observable.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str
        Pattern of entanglement: "circular", "full", or "none".

    Returns
    -------
    qml.QNode
        A callable quantum node that accepts a vector of input features and
        returns the expectation value of a Pauli operator.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Encode inputs via RY rotations
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)

        # Entanglement layer
        if entanglement == "circular":
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qml.CNOT(wires=[i, j])

        # Parameterised rotations
        qml.layer(qml.Rot, wires=range(num_qubits), params=weights)

        # Observable: tensor product of Pauli Z on all qubits
        return qml.expval(qml.PauliZ(wires=range(num_qubits)).tensor_product(
            *[qml.PauliZ(wires=[i]) for i in range(num_qubits)]
        ))

    return circuit


__all__ = ["EstimatorQNN"]
