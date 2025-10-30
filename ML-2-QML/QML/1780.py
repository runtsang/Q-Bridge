"""Quantum QCNN implemented with PennyLane.

The circuit uses an entangled Z‑feature map followed by a layered ansatz
consisting of parameterised rotations and CNOT entangling gates.  The
output is a single‑qubit expectation value of the Pauli‑Z operator,
which can be used as a classical log‑likelihood in hybrid training.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane import qnn


def _z_feature_map(num_wires: int) -> qml.Device:
    """Entangled Z‑feature map using linear CNOT connections."""
    return qml.templates.embeddings.ZFeatureMap(num_wires=num_wires, entanglement='linear')


def _ansatz(num_wires: int, params: np.ndarray) -> None:
    """Layered ansatz: RY‑RZ rotations followed by CNOT entanglement."""
    param_idx = 0
    for layer in range(3):
        # Rotations
        for w in range(num_wires):
            qml.RZ(params[param_idx], wires=w)
            param_idx += 1
            qml.RY(params[param_idx], wires=w)
            param_idx += 1
        # Entanglement
        for w in range(0, num_wires - 1):
            qml.CNOT(wires=[w, w + 1])
        # Wrap around
        qml.CNOT(wires=[num_wires - 1, 0])


def QCNN() -> qml.QNode:
    """Return a PennyLane QNode representing the quantum QCNN.

    The QNode outputs the expectation value of Pauli‑Z on the last qubit.
    """
    num_wires = 8
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs: np.ndarray, params: np.ndarray) -> float:
        # Feature map
        _z_feature_map(num_wires)(inputs)
        # Ansatz
        _ansatz(num_wires, params)
        # Measurement
        return qml.expval(qml.PauliZ(num_wires - 1))

    return circuit


__all__ = ["QCNN"]
