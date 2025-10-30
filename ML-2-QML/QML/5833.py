"""Quantum self‑attention module using PennyLane variational circuits.

The circuit implements a parameterised rotation on each qubit followed
by pairwise controlled‑RX gates that emulate the “entangle” step of the
classical self‑attention block.  The measurement outcomes are converted
into a probability vector that is interpreted as the attention output.

Parameters
----------
n_qubits : int
    Number of qubits; this also determines the dimensionality of the
    input embeddings.
dev : pennylane.Device, optional
    PennyLane device; if None a default qml.device('default.qubit',
    wires=n_qubits) is created.
"""

import numpy as np
import pennylane as qml
import torch

class SelfAttentionGen231:
    """Quantum self‑attention using PennyLane variational circuits."""
    def __init__(self, n_qubits: int, dev: qml.Device = None):
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(rotation_params, entangle_params, inputs):
            # Encode the inputs as a product state
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)

            # Rotation layer
            for i in range(n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement layer
            for i in range(n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Measure expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return a probability distribution that
        mimics the attention output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of length ``3 * n_qubits``.
        entangle_params : np.ndarray
            Flat array of length ``n_qubits - 1``.
        inputs : np.ndarray
            Array of length ``n_qubits`` representing the input embedding.
        shots : int, optional
            Number of shots for the measurement; if shots is None, the
            expectation values are returned directly.

        Returns
        -------
        np.ndarray
            Probability vector of length ``n_qubits``.
        """
        if rotation_params.size!= 3 * self.n_qubits:
            raise ValueError("rotation_params size mismatch")
        if entangle_params.size!= self.n_qubits - 1:
            raise ValueError("entangle_params size mismatch")
        if inputs.size!= self.n_qubits:
            raise ValueError("inputs size mismatch")

        if shots is None:
            # Expectation values
            probs = self.circuit(rotation_params, entangle_params, inputs)
        else:
            # Sampling
            results = self.dev.execute(
                self.circuit, shots=shots,
                rotation_params=rotation_params,
                entangle_params=entangle_params,
                inputs=inputs
            )
            probs = results[0]
            # Convert counts to probabilities
            probs = {k: v / shots for k, v in probs.items()}

        # Convert to a vector of probabilities for each qubit
        prob_vector = np.zeros(self.n_qubits)
        for state, prob in probs.items():
            for i, bit in enumerate(state):
                if bit == '1':
                    prob_vector[i] += prob
        return prob_vector

__all__ = ["SelfAttentionGen231"]
