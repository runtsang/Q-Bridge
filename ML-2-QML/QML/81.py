"""Quantum sampler network with a variational ansatz and probability extraction."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import QNode


class SamplerQNNGen089:
    """
    Variational quantum sampler network that extends the original design.
    Uses Pennylane to construct a parameterized circuit with RY rotations
    and CNOT entangling gates. Provides methods to compute probability
    distributions and sample from the circuit.

    Parameters
    ----------
    n_qubits : int, default 2
        Number of qubits in the circuit.
    layers : int, default 2
        Number of variational layers (each layer contains RY on all qubits
        followed by a CNOT ladder).
    device : str or qml.Device, optional
        Pennylane device name or instance. Defaults to the Qiskit simulator.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        layers: int = 2,
        device: qml.Device | str = "default.qubit",
    ) -> None:
        if isinstance(device, str):
            self.dev = qml.device(device, wires=n_qubits)
        else:
            self.dev = device

        self.n_qubits = n_qubits
        self.layers = layers
        # Total number of variational parameters: n_qubits * layers
        self.params = np.random.uniform(
            low=0.0, high=2 * np.pi, size=(layers, n_qubits)
        )

    def circuit(self, inputs: np.ndarray, weights: np.ndarray) -> None:
        """Parameterized circuit used by the QNode."""
        for i in range(self.layers):
            for q in range(self.n_qubits):
                qml.RY(inputs[i, q] if inputs is not None else 0.0, wires=q)
                qml.RY(weights[i, q], wires=q)
            # CNOT ladder
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

    def _qnode(self) -> QNode:
        @qml.qnode(self.dev, interface="autograd")
        def sampler_qnode(inputs, weights):
            self.circuit(inputs, weights)
            return qml.probs(wires=range(self.n_qubits))

        return sampler_qnode

    def probabilities(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution over all basis states.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters of shape (layers, n_qubits).
        weights : np.ndarray
            Weight parameters of shape (layers, n_qubits).

        Returns
        -------
        np.ndarray
            Probability vector of length 2**n_qubits.
        """
        qnode = self._qnode()
        probs = qnode(inputs, weights)
        return probs

    def sample(self, inputs: np.ndarray, weights: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """
        Draw samples from the quantum circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters.
        weights : np.ndarray
            Weight parameters.
        n_samples : int, default 1000
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing sampled bitstrings.
        """
        probs = self.probabilities(inputs, weights)
        # Convert probabilities to cumulative distribution
        cum_probs = np.cumsum(probs)
        rng = np.random.default_rng()
        uniform_samples = rng.uniform(size=n_samples)
        indices = np.searchsorted(cum_probs, uniform_samples)
        # Convert indices to bitstrings
        bitstrings = [(np.binary_repr(idx, width=self.n_qubits)) for idx in indices]
        return np.array(bitstrings)

    def __repr__(self) -> str:
        return f"<SamplerQNNGen089 n_qubits={self.n_qubits} layers={self.layers}>"

__all__ = ["SamplerQNNGen089"]
