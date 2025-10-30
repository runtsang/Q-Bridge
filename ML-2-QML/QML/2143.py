"""Core circuit factory for the incremental data‑uploading classifier using PennyLane."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
from pennylane import numpy as np


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    device_name: str = "default.qubit",
    shots: int = 1000,
) -> Tuple[qml.QNode, Iterable, Iterable, List[qml.operation.Operator]]:
    """
    Construct a PennyLane variational circuit with data‑encoding, trainable layers,
    and Z‑observables on each qubit. The function returns a differentiable QNode,
    the parameter names, and the measurement operators, mirroring the
    classical API signature.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    device_name : str, optional
        PennyLane backend name (default: ``default.qubit``).
    shots : int, optional
        Number of shots for expectation estimation.

    Returns
    -------
    Tuple[qml.QNode, Iterable, Iterable, List[qml.operation.Operator]]
    """
    dev = qml.device(device_name, wires=num_qubits, shots=shots)

    # Parameter names for easier debugging
    encoding = [f"x_{i}" for i in range(num_qubits)]
    weights = [f"theta_{d}_{i}" for d in range(depth) for i in range(num_qubits)]

    @qml.qnode(dev)
    def circuit(*params):
        # Data‑encoding layer
        for i, x in enumerate(params[:num_qubits]):
            qml.RX(x, wires=i)

        # Variational layers
        offset = num_qubits
        for d in range(depth):
            for i in range(num_qubits):
                qml.RY(params[offset + d * num_qubits + i], wires=i)
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        # Measurement: expectation value of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    observables = [qml.PauliZ(i) for i in range(num_qubits)]
    return circuit, encoding, weights, observables


__all__ = ["build_classifier_circuit"]
