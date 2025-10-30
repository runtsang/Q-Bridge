"""Quantum sampler using Pennylane with adaptive measurement."""
from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Tuple

__all__ = ["SamplerQNN"]

def SamplerQNN() -> qml.QNode:
    """
    Returns a Pennylane QNode that implements a 3‑qubit variational sampler.
    The circuit consists of parameterized Ry rotations, a full‑connect
    entangling layer, and a measurement of all qubits in the computational basis.
    The function also attaches a ``sample`` method to the QNode for easy
    probability extraction and sampling via the default simulator.
    """
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Input encoding: Ry rotations
        for i, w in enumerate(inputs):
            qml.RY(w, wires=i)

        # Entangling layer
        for i in range(3):
            qml.CNOT(wires=[i, (i + 1) % 3])

        # Parameterized rotations
        for i, w in enumerate(weights):
            qml.RY(w, wires=i)

        # Measurement in computational basis
        return qml.probs(wires=range(3))

    # Attach a sampling helper to the QNode
    def sample(inputs: np.ndarray, n_samples: int = 1) -> np.ndarray:
        probs = circuit(inputs, np.zeros(3))
        return np.random.choice(8, size=n_samples, p=probs)

    circuit.sample = sample
    return circuit
