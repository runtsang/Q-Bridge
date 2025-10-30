"""Pennylane implementation of a data‑re‑uploading variational classifier.

The circuit mirrors the classical architecture: encoding, a stack of
parameterized rotation layers, and entanglement.  It supports
parameter‑shift gradient evaluation and returns expectation values
of local Z observables, which can be summed to form the classification
score.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
from pennylane import numpy as np


def _entangle_layer(qubits: List[int]) -> None:
    """Apply a ring of CZ gates."""
    for i in range(len(qubits) - 1):
        qml.CZ(qubits[i], qubits[i + 1])
    qml.CZ(qubits[-1], qubits[0])  # periodic boundary


def _rotation_layer(params: np.ndarray, qubits: List[int]) -> None:
    """Apply a single‑qubit rotation layer."""
    for q, theta in zip(qubits, params):
        qml.RZ(theta, q)


def _data_encoding(x: np.ndarray, qubits: List[int]) -> None:
    """Encode classical data via Rx rotations."""
    for q, val in zip(qubits, x):
        qml.RX(val, q)


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[qml.QNode, Iterable[int], Iterable[int], List[qml.operation.Operator]]:
    """
    Build a Pennylane QNode representing a data‑re‑uploading classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : qml.QNode
        The callable quantum circuit returning expectation values.
    encoding : Iterable[int]
        Indices of input features used for encoding.
    weights : Iterable[int]
        Linear indices of variational parameters.
    observables : List[qml.operation.Operator]
        Local Z observables for each qubit.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Return expectation values of local Z observables."""
        _data_encoding(x, range(num_qubits))
        for d in range(depth):
            _rotation_layer(params[d * num_qubits : (d + 1) * num_qubits], range(num_qubits))
            _entangle_layer(range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    encoding = list(range(num_qubits))
    weights = list(range(num_qubits * depth))
    observables = [qml.PauliZ(i) for i in range(num_qubits)]
    return circuit, encoding, weights, observables


__all__ = ["build_classifier_circuit"]
