"""Quantum sampler network using PennyLane."""

import numpy as np
import pennylane as qml


class SamplerQNN:
    """
    Variational quantum sampler that models a probability distribution over four
    computational‑basis states. The circuit is parameterized by ``weights`` (rotation
    angles) and ``inputs`` (classical data encoded via Ry gates). Sampling is performed
    on a default.qubit device, and a simple gradient estimate is supplied for
    supervised training.
    """

    def __init__(self, device_name: str = "default.qubit", wires: int = 2) -> None:
        """
        Parameters
        ----------
        device_name : str
            PennyLane device to use (e.g., 'default.qubit', 'qiskit.aer', etc.).
        wires : int
            Number of qubits in the circuit.
        """
        self.dev = qml.device(device_name, wires=wires)
        self.weights = np.random.uniform(0, 2 * np.pi, size=4)
        self.inputs = np.zeros(2)

        # Build the QNode once; it will be called with new parameters each time.
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Define the variational circuit and return measurement probabilities."""
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)
        return qml.probs(wires=[0, 1])

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution for a single classical input.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (2,) array of classical features.

        Returns
        -------
        np.ndarray
            Probabilities for the four computational‑basis states.
        """
        return self.qnode(inputs, self.weights)

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the quantum distribution.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (2,) array of classical features.
        num_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Integer samples of shape (num_samples,) with values 0‑3.
        """
        probs = self.evaluate(inputs)
        return np.random.choice(len(probs), size=num_samples, p=probs)

    def set_weights(self, weights: np.ndarray) -> None:
        """Update the circuit parameters (useful for training)."""
        self.weights = np.asarray(weights)

    def gradient(self, inputs: np.ndarray, target: int) -> np.ndarray:
        """
        Compute a simple gradient estimate using the parameter‑shift rule for a
        single target class.

        Parameters
        ----------
        inputs : np.ndarray
            Classical input.
        target : int
            Index of the desired outcome (0‑3).

        Returns
        -------
        np.ndarray
            Gradient vector of shape (4,).
        """
        def loss_fn(w):
            probs = self.qnode(inputs, w)
            return -np.log(probs[target] + 1e-12)

        return qml.grad(loss_fn)(self.weights)


__all__ = ["SamplerQNN"]
