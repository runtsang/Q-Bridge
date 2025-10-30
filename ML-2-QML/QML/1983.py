"""Quantum estimator with entangling layers and multi‑observable readout.

The module exposes `EstimatorQNNGen181`, a function that builds a
Pennylane QNode.  The circuit processes two classical inputs and two
trainable weight parameters.  It uses a stack of CNOT‑entangling layers
followed by parameterised rotations, and returns a weighted sum of
Pauli‑Z and Pauli‑X expectation values as the output.
"""

import pennylane as qml
import pennylane.numpy as np
import torch


def EstimatorQNNGen181(num_qubits: int = 3, layers: int = 2):
    """Return a Pennylane QNode estimator.

    Args:
        num_qubits: Number of qubits in the device.
        layers: Number of entangling‑rotation layers.

    Returns:
        A PennyLane QNode that can be used as a hybrid layer in a
        Torch‑based training loop.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Encode two classical features into Y‑rotations
        for i in range(num_qubits):
            qml.RY(inputs[i % 2], wires=i)

        # Entangling‑rotation layers
        for _ in range(layers):
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.layer(qml.RY, wires=range(num_qubits))

        # Weighted readout of Pauli expectations
        out_z = qml.expval(qml.PauliZ(0))
        out_x = qml.expval(qml.PauliX(1))
        return out_z * weights[0] + out_x * weights[1]

    return circuit


__all__ = ["EstimatorQNNGen181"]
