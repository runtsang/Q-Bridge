"""Quantum sampler network using Pennylane variational circuit."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.qnode import QNode


class SamplerQNN:
    """
    Hybrid quantum sampler.

    The circuit consists of two layers of parameterized Ry rotations followed by a
    CNOT entangling gate and a second layer of Ry rotations.  The input parameters
    encode the two features, while the weight parameters are trainable.

    Attributes
    ----------
    n_qubits : int
        Number of qubits (fixed to 2 for this sampler).
    dev : qml.Device
        Quantum device used for simulation.
    weight_params : np.ndarray
        Trainable weight parameters.
    """

    def __init__(self, dev: qml.Device | None = None) -> None:
        self.n_qubits = 2
        self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)
        # 8 weight parameters: 4 per layer
        self.weight_params = pnp.random.uniform(low=0, high=2 * np.pi, size=(8,), requires_grad=True)

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Parameterized quantum circuit."""
        # Layer 1
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        # Weight layer
        for i in range(4):
            qml.RY(weights[i], wires=i % 2)
        qml.CNOT(wires=[0, 1])
        # Layer 2
        for i in range(4, 8):
            qml.RY(weights[i], wires=(i - 4) % 2)
        # Measurement
        return qml.expval(qml.PauliZ(0))

    @QNode
    def qnode(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self._circuit(inputs, weights)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the sampler on a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch_size, 2) containing two features per sample.

        Returns
        -------
        np.ndarray
            Probabilities obtained from the quantum circuit, normalized over the batch.
        """
        probs = []
        for inp in inputs:
            val = self.qnode(inp, self.weight_params)
            probs.append(val)
        probs = np.array(probs)
        # Convert expectation values to probabilities via softmax
        return np.exp(probs) / np.sum(np.exp(probs))

    def get_params(self) -> dict:
        """Return a dictionary of trainable parameters."""
        return {"weight_params": self.weight_params.detach().numpy()}


__all__ = ["SamplerQNN"]
