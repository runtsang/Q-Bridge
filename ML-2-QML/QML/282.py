"""Quantum neural network with entanglement and adjustable parameters.

The EstimatorQNN function returns a Pennylane QNode that implements a
two‑qubit variational circuit.  Inputs are encoded with an embedding
gate, followed by two layers of parameterised rotations and a CNOT
entangling gate.  The circuit is callable with torch tensors and
produces the expectation value of Pauli‑Z on qubit 0, making it suitable
for integration with classical optimisers.
"""

import pennylane as qml
import pennylane.numpy as np
from pennylane import torch as ptr
from torch import Tensor


dev = qml.device("default.qubit", wires=2)


def EstimatorQNN() -> qml.QNode:
    """Return a Pennylane QNode implementing a 2‑qubit variational circuit."""

    @qml.qnode(dev, interface="torch")
    def circuit(params: Tensor, inputs: Tensor) -> Tensor:
        # Encode 2‑dimensional classical data into qubits
        qml.Embedding(inputs, wires=range(2), normalize=False)

        # Variational layers
        for i in range(2):
            qml.RY(params[0, i], wires=i)
            qml.RZ(params[1, i], wires=i)
            qml.CNOT(wires=[i, (i + 1) % 2])

        # Measurement: expectation value of Pauli‑Z on qubit 0
        return qml.expval(qml.PauliZ(0))

    return circuit
