"""SamplerQNN: Quantum variational sampler with sampling and gradient support.

This module implements a 2‑qubit variational circuit that maps a 2‑dimensional
classical input to a probability distribution over the computational basis.
The circuit is parameterised with input angles and trainable weights.  The
class exposes a `forward` method that returns the probability vector and a
`sample` method that draws samples from the distribution.  Gradients are
obtained via the parameter‑shift rule provided by Pennylane.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Tuple

class SamplerQNN:
    """Quantum sampler using a 2‑qubit variational circuit."""

    def __init__(self,
                 dev: qml.Device = None,
                 hidden_layers: int = 2) -> None:
        self.num_qubits = 2
        self.hidden_layers = hidden_layers
        self.dev = dev or qml.device("default.qubit", wires=self.num_qubits)
        # Trainable parameters: shape (hidden_layers, num_qubits)
        self.weights = pnp.random.uniform(0, 2 * np.pi,
                                          size=(hidden_layers, self.num_qubits))

    def _circuit(self, inputs: Tuple[float, float], weights: np.ndarray) -> np.ndarray:
        """Variational circuit that returns the probability of each basis state."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit_qnode(inp, w):
            # Encode inputs as Ry rotations
            for i in range(self.num_qubits):
                qml.RY(inp[i], wires=i)
            # Variational layers
            for layer in range(self.hidden_layers):
                for i in range(self.num_qubits):
                    qml.RY(w[layer, i], wires=i)
                # Simple CNOT entanglement
                qml.CNOT(0, 1)
            return qml.probs(wires=range(self.num_qubits))

        return circuit_qnode(inputs, weights)

    def forward(self,
                inputs: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
        """
        Compute probability distribution for each input in batch.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (..., 2) with input angles in [0, 2π].
        weights : np.ndarray
            Shape (hidden_layers, 2) trainable parameters.

        Returns
        -------
        probs : np.ndarray
            Shape (..., 4) probability of each 2‑qubit basis state.
        """
        probs = np.array([self._circuit(inp, weights) for inp in inputs])
        return probs

    def sample(self,
               inputs: np.ndarray,
               weights: np.ndarray,
               n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the quantum circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (..., 2) input angles.
        weights : np.ndarray
            Trainable parameters.
        n_samples : int
            Number of samples per input.

        Returns
        -------
        samples : np.ndarray
            Array of shape (..., n_samples, num_qubits) containing bitstrings.
        """
        @qml.qnode(self.dev, interface="autograd")
        def sample_qnode(inp, w):
            # Encode inputs
            for i in range(self.num_qubits):
                qml.RY(inp[i], wires=i)
            # Variational layers
            for layer in range(self.hidden_layers):
                for i in range(self.num_qubits):
                    qml.RY(w[layer, i], wires=i)
                qml.CNOT(0, 1)
            return qml.sample(wires=range(self.num_qubits))

        samples = np.array([sample_qnode(inp, weights) for inp in inputs])
        return samples

    def loss(self,
             inputs: np.ndarray,
             targets: np.ndarray,
             weights: np.ndarray) -> float:
        """
        Cross‑entropy loss between predicted probabilities and one‑hot targets.

        Parameters
        ----------
        inputs : np.ndarray
            Input angles.
        targets : np.ndarray
            One‑hot encoded target distribution of shape (..., 4).
        weights : np.ndarray
            Trainable parameters.

        Returns
        -------
        loss : float
            Mean cross‑entropy over the batch.
        """
        probs = self.forward(inputs, weights)
        ce = -np.sum(targets * np.log(probs + 1e-12), axis=-1)
        return np.mean(ce)

    def gradient(self,
                 inputs: np.ndarray,
                 targets: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. trainable weights using parameter‑shift.

        Parameters
        ----------
        inputs : np.ndarray
            Input angles.
        targets : np.ndarray
            One‑hot targets.
        weights : np.ndarray
            Current weights.

        Returns
        -------
        grad : np.ndarray
            Gradient array of same shape as weights.
        """
        def loss_fn(w):
            return self.loss(inputs, targets, w)

        grad = qml.grad(loss_fn)(weights)
        return grad

__all__ = ["SamplerQNN"]
