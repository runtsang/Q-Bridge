"""Quantum hybrid estimator based on a 2‑qubit variational circuit.

The EstimatorQNN class encapsulates a Pennylane QNode that
parameterises a shallow circuit with alternating rotation and
entangling layers.  The circuit returns the expectation value
of the Pauli‑Z operator on the first qubit, which serves as the
regression output.  The interface mirrors the classical
EstimatorQNN for easy comparison.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class EstimatorQNN:
    """Variational quantum circuit for regression.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    layers : int, default 2
        Number of alternating rotation–entanglement blocks.
    dev_name : str, default "default.qubit"
        Pennylane device name.  Use "qiskit.ibmq" for real hardware.
    """
    def __init__(
        self,
        num_qubits: int = 2,
        layers: int = 2,
        dev_name: str = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device(dev_name, wires=self.num_qubits)

        # Build the QNode once; interface='torch' keeps tensors as torch
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, *weights_and_inputs: pnp.ndarray) -> pnp.ndarray:
        """Variational circuit.

        The first ``num_qubits`` parameters are treated as input
        embeddings (via RY rotations).  The remaining parameters
        are trainable weights.  The circuit returns the expectation
        of the Pauli‑Z operator on qubit 0.
        """
        # Unpack inputs and weights
        inputs = weights_and_inputs[: self.num_qubits]
        weights = weights_and_inputs[self.num_qubits :]

        # Input encoding
        for i, inp in enumerate(inputs):
            qml.RY(inp, wires=i)

        # Variational layers
        weight_iter = iter(weights)
        for _ in range(self.layers):
            for i in range(self.num_qubits):
                qml.RX(next(weight_iter), wires=i)
                qml.RZ(next(weight_iter), wires=i)
            # Entangling
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Measurement
        return qml.expval(qml.PauliZ(0))

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Evaluate the circuit.

        Parameters
        ----------
        inputs
            Array of shape ``(num_qubits,)`` containing the input
            features to embed via RY rotations.
        weights
            Flattened array of trainable parameters.  Its length must
            equal ``layers * 2 * num_qubits``.
        """
        return self.qnode(*inputs, *weights)

__all__ = ["EstimatorQNN"]
