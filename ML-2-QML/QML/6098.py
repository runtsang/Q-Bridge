#!/usr/bin/env python
"""Enhanced EstimatorQNN: a 2‑qubit variational quantum circuit."""
from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=2)

def feature_map(x):
    """Encode 2‑dimensional input into two qubits."""
    qml.RY(x[0], wires=0)
    qml.RZ(x[1], wires=1)
    qml.CNOT(wires=[0, 1])

def variational_circuit(x, weights):
    feature_map(x)
    for i, w in enumerate(weights):
        qml.RY(w, wires=i)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

def EstimatorQNN(**kwargs):
    """Return a Pennylane QNode for quantum regression."""
    hidden_layers = kwargs.get("hidden_layers", 1)
    weights = [np.random.randn(2) for _ in range(hidden_layers)]
    return qml.QNode(variational_circuit, dev, interface="autograd")

__all__ = ["EstimatorQNN"]
