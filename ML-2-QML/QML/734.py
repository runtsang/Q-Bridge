"""
`SamplerQNN` – A parameterised quantum sampler.

Enhancements over the seed:
* Uses PennyLane’s declarative QNode for easy integration with hybrid training.
* Supports an arbitrary number of qubits and entangling layers.
* Provides both shot‑based sampling (useful for noisy hardware) and state‑vector probability extraction.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Callable, Iterable, Sequence


class SamplerQNN:
    """
    Quantum sampler that maps a 2‑dimensional classical input to a probability distribution
    over measurement outcomes of a register of `num_qubits` qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the sampler circuit.
    entangle_layers : int, optional
        Number of alternating CX‑entangling layers after each rotation block.
    seed : int | None, optional
        Random seed for reproducible device initialization.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        entangle_layers: int = 1,
        seed: int | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.entangle_layers = entangle_layers
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=None, seed=seed)

        # Parameter placeholders
        self.input_params = qml.numpy.array([0.0] * 2)
        self.weight_params = qml.numpy.array([0.0] * (4 * entangle_layers))

        # Build the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, *params) -> np.ndarray:
        """Variational circuit with input encoding, entanglement, and rotation layers."""
        # Unpack parameters
        inputs = params[:2]
        weights = params[2:]

        # Input‑encoding: Ry rotations
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)

        # Entangling + rotation layers
        for l in range(self.entangle_layers):
            # Entanglement block
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Rotation block
            for w, i in zip(weights[l * self.num_qubits : (l + 1) * self.num_qubits], range(self.num_qubits)):
                qml.RY(w, wires=i)

        # Measurement: return probability amplitude vector
        return qml.state()

    def probabilities(self, inputs: Iterable[float] | np.ndarray) -> np.ndarray:
        """
        Compute the full probability distribution over basis states for a given input.

        Parameters
        ----------
        inputs : array-like
            Two‑dimensional input vector to encode.

        Returns
        -------
        np.ndarray
            Probability vector of shape ``(2**num_qubits,)``.
        """
        probs = np.abs(self.qnode(*np.concatenate([inputs, self.weight_params]))) ** 2
        return probs

    def sample(self, inputs: Iterable[float] | np.ndarray, n_shots: int = 1000) -> np.ndarray:
        """
        Draw samples from the measurement distribution using a shot‑based run.

        Parameters
        ----------
        inputs : array-like
            Two‑dimensional input vector to encode.
        n_shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_shots,)`` containing basis‑state indices.
        """
        # Re‑instantiate a device with shots for sampling
        samp_dev = qml.device("default.qubit", wires=self.num_qubits, shots=n_shots)
        samp_qnode = qml.QNode(self._circuit, samp_dev)
        result = samp_qnode(*np.concatenate([inputs, self.weight_params]))
        return result

    def set_weights(self, weights: Sequence[float]) -> None:
        """Convenience setter for the trainable weight parameters."""
        self.weight_params = np.array(weights)

    def get_weights(self) -> np.ndarray:
        """Return the current weight parameters."""
        return self.weight_params


__all__ = ["SamplerQNN"]
