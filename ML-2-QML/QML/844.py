"""Quantum sampler network implemented with PennyLane.

The circuit consists of two qubits with parameterised Ry rotations
followed by a CNOT entangling gate.  The sampler returns the probability
distribution over the computational basis, which can be used directly
as a quantum analogue of the classical softmax output.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Tuple


class SamplerQNN:
    """
    Variational sampler that produces a 2‑dimensional probability vector.
    Parameters
    ----------
    num_qubits : int
        Number of qubits (fixed to 2 for this seed).
    init_weights : Tuple[float, float, float, float] | None
        Optional initial weights for the Ry rotations.
    """

    def __init__(self, num_qubits: int = 2,
                 init_weights: Tuple[float, float, float, float] | None = None) -> None:
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Initialise trainable parameters
        if init_weights is None:
            init_weights = np.random.uniform(0, 2 * np.pi, size=4)
        self.weights = np.array(init_weights, requires_grad=True)

        @qml.qnode(self.dev)
        def circuit(inputs: np.ndarray):
            # Input encoding
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Variational block
            qml.RY(self.weights[0], wires=0)
            qml.RY(self.weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(self.weights[2], wires=0)
            qml.RY(self.weights[3], wires=1)

            # Measurement: return probabilities of |00>, |01>, |10>, |11>
            return qml.probs(wires=range(num_qubits))

        self._circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution for a single input.
        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (2,) containing the input features.
        Returns
        -------
        probs : np.ndarray
            Array of shape (2,) containing the marginal probabilities
            for qubit 0 (|0> vs |1>) after tracing out qubit 1.
        """
        full_probs = self._circuit(inputs)
        # Collapse to the first qubit by summing over the second qubit
        probs_q0 = np.array([full_probs[0] + full_probs[2],  # |0> on qubit 0
                             full_probs[1] + full_probs[3]])  # |1> on qubit 0
        return probs_q0

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Negative log‑likelihood loss between predicted probabilities and targets.
        """
        preds = self.forward(inputs)
        # Avoid log(0) by clipping
        eps = 1e-10
        return -float(np.sum(targets * np.log(preds + eps)))

    def train_step(self, inputs: np.ndarray, targets: np.ndarray,
                   lr: float = 0.01) -> None:
        """
        Perform a single gradient‑descent step on the variational parameters.
        """
        grads = qml.grad(self.loss)(inputs, targets)
        self.weights = self.weights - lr * grads

    def predict(self, batch_inputs: np.ndarray) -> np.ndarray:
        """
        Batch prediction over multiple inputs.
        """
        return np.array([self.forward(x) for x in batch_inputs])

__all__ = ["SamplerQNN"]
