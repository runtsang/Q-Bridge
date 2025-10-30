"""Quantum classifier builder employing a variational ansatz and amplitude encoding."""
from __future__ import annotations
from typing import Iterable, Tuple
import pennylane as qml
from pennylane.pauli import PauliZ
import numpy as np

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[qml.QNode, Iterable, Iterable, list[PauliZ]]:
    """Return a qnode, encoding parameters, variational parameters and measurement observables."""
    dev = qml.device("default.qubit", wires=num_qubits)

    # encoding parameters (data re-uploading)
    encoding = [f"x_{i}" for i in range(num_qubits)]

    # variational parameters
    theta = [f"theta_{d}_{q}" for d in range(depth) for q in range(num_qubits)]

    @qml.qnode(dev, interface="jax")
    def circuit(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        # amplitude encoding of input features
        qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True)

        # variational layers
        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
            # entangling pattern: ring of CZ gates
            for q in range(num_qubits):
                qml.CZ(wires=[q, (q + 1) % num_qubits])

        # measurement: expectation of Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    observables = [PauliZ(i) for i in range(num_qubits)]
    return circuit, encoding, theta, observables

__all__ = ["build_classifier_circuit"]
