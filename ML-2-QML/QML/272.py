"""Quantum sampler implemented with Pennylane.

The SamplerQNN class defines a variational circuit on two qubits with
parameterised rotations and an entangling layer.  The ``sample`` method
returns measurement outcomes sampled from the circuit probability
distribution, matching the API of the classical version.
"""

import pennylane as qml
import numpy as np

class SamplerQNN:
    """Variational quantum sampler on two qubits."""

    def __init__(self, num_qubits: int = 2, num_layers: int = 2, seed: int | None = None):
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits (default 2).
        num_layers : int
            Number of rotation-entanglement layers.
        seed : int | None
            Random seed for initial parameters.
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = self._init_params(seed)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Input encoding: Ry rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(num_layers):
                for qubit in range(num_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                # Entanglement
                for qubit in range(num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[num_qubits - 1, 0])  # wrap‑around
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def _init_params(self, seed: int | None):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(self.num_layers, self.num_qubits, 2))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape ``(batch, 2)``.
        Returns
        -------
        np.ndarray
            Probability vector of shape ``(batch, 4)`` for the four basis states.
        """
        return np.array([self.circuit(input, self.params) for input in inputs])

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Sample measurement outcomes from the circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Shape ``(batch, 2)``.
        num_samples : int
            Number of samples per input.

        Returns
        -------
        np.ndarray
            Samples of shape ``(batch, num_samples, num_qubits)`` with values 0 or 1.
        """
        probs = self.forward(inputs)
        samples = []
        for p in probs:
            idxs = np.random.choice(len(p), size=num_samples, p=p)
            bits = np.array([[ (idx >> i) & 1 for i in range(self.num_qubits)] for idx in idxs])
            samples.append(bits)
        return np.array(samples)

__all__ = ["SamplerQNN"]
