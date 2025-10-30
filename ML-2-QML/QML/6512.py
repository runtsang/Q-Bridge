"""Core circuit factory for the incremental data‑uploading classifier with PennyLane."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import pennylane.numpy as pnp


def build_classifier_circuit(num_qubits: int, depth: int, *,
                             device: str = "default.qubit",
                             shots: int = 1024) -> Tuple[qml.QNode, Iterable[pnp.ndarray], Iterable[pnp.ndarray], List[qml.operation.Operator]]:
    """
    Construct a variational circuit with explicit data encoding and parameterised ansatz.

    Parameters
    ----------
    num_qubits:
        Number of qubits, equal to the feature dimension.
    depth:
        Number of variational layers.
    device:
        PennyLane device name (defaults to the simulator).
    shots:
        Number of measurement shots for expectation evaluation.

    Returns
    -------
    circuit:
        ``qml.QNode`` that returns a vector of expectation values.
    encoding:
        Vector of data‑encoding parameters (identity mapping).
    weights:
        Variational parameters ready for optimisation.
    observables:
        List of Pauli‑Z observables, one per qubit.
    """
    dev = qml.device(device, wires=num_qubits, shots=shots)

    # Identity encoding – the circuit will be supplied with the actual data at runtime.
    encoding = pnp.array([0.0] * num_qubits)
    weights = pnp.array([0.0] * num_qubits * depth)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: pnp.ndarray, params: pnp.ndarray) -> pnp.ndarray:
        # Data encoding
        for i, w in enumerate(inputs):
            qml.RX(w, wires=i)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for i in range(num_qubits):
                qml.RY(params[idx], wires=i)
                idx += 1
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    observables = [qml.PauliZ(i) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
