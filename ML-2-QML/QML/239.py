"""Quantum sampler network using PennyLane. Provides a parameterised circuit that outputs
a probability distribution over two outcomes. Includes sampling and gradient
evaluation utilities."""
import pennylane as qml
import numpy as np
from typing import Tuple


class SamplerQNN:
    """
    Variational quantum circuit that samples from a 2‑qubit system.
    The circuit consists of input rotations, a CX entanglement layer,
    and trainable Ry rotations. Probabilities are obtained from the
    computational basis measurement.
    """

    def __init__(
        self,
        dev: qml.Device | None = None,
        num_qubits: int = 2,
        init_weights: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.dev = dev or qml.device("default.qubit", wires=num_qubits)
        self.input_params = qml.numpy.array([0.0] * num_qubits, requires_grad=True)
        self.weight_params = qml.numpy.array(
            init_weights if init_weights is not None else np.random.randn(2 * num_qubits),
            requires_grad=True,
        )

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            """Variational circuit parameterised by inputs and weights."""
            # Input encoding
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            # Trainable rotations
            for i in range(num_qubits):
                qml.RY(weights[i], wires=i)
            qml.CNOT(wires=[0, 1])
            for i in range(num_qubits):
                qml.RY(weights[num_qubits + i], wires=i)
            # Measurement
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution for a single input vector.

        Parameters
        ----------
        inputs : array‑like
            Input vector of shape (num_qubits,).

        Returns
        -------
        np.ndarray
            Probability distribution over 2^num_qubits basis states.
        """
        probs = self.circuit(inputs, self.weight_params)
        return np.array(probs)

    def sample(self, inputs: np.ndarray, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        """
        Draw samples from the circuit output distribution.

        Parameters
        ----------
        inputs : array‑like
            Input vector of shape (num_qubits,).
        n_samples : int
            Number of samples to draw.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Sample indices in the computational basis.
        """
        probs = self.evaluate(inputs)
        rng = np.random.default_rng(seed)
        return rng.choice(len(probs), size=n_samples, p=probs)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current trainable parameters (weights)."""
        return self.weight_params

    def set_params(self, new_weights: np.ndarray) -> None:
        """Set new trainable weights."""
        self.weight_params = new_weights

    def gradient(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the probability distribution w.r.t. the trainable weights.

        Parameters
        ----------
        inputs : array‑like
            Input vector.

        Returns
        -------
        np.ndarray
            Gradient array of shape (len(weight_params), 2**num_qubits).
        """
        grads = qml.grad(self.circuit)(inputs, self.weight_params)
        return grads
