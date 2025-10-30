"""
Quantum sampler network employing a variational circuit and state‑vector sampling.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane import device as qml_device
from pennylane.optimize import AdamOptimizer
from typing import Tuple


class SamplerQNN:
    """
    Variational sampler that maps two classical inputs to a probability
    distribution over the computational basis states of two qubits.

    Parameters
    ----------
    dev_name : str, optional
        The Pennylane device to use. Default is "default.qubit".
    dev_qubits : int, optional
        Number of qubits in the device. Default is 2.
    """

    def __init__(self, dev_name: str = "default.qubit", dev_qubits: int = 2) -> None:
        self.dev = qml_device(dev_name, wires=dev_qubits)
        self.params = np.random.uniform(0, 2 * np.pi, size=(4,))
        self.input_params = np.arange(2)
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev)
        def circuit(inputs: Tuple[float, float], weights: np.ndarray) -> np.ndarray:
            # Parameterized rotations from classical inputs
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Entanglement layer
            qml.CNOT(wires=[0, 1])

            # Variational layer
            for i, w in enumerate(weights):
                qml.RY(w, wires=i)

            # Second entanglement
            qml.CNOT(wires=[0, 1])

            # Measure in computational basis
            return qml.probs(wires=[0, 1])

        self._circuit = circuit

    def forward(
        self,
        inputs: Tuple[float, float],
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the probability distribution for given inputs and weights.

        Parameters
        ----------
        inputs : tuple[float, float]
            Two classical input values.
        weights : np.ndarray, optional
            Parameter vector of length 4. If None, the current parameters are used.

        Returns
        -------
        probs : np.ndarray
            Array of shape (4,) with probabilities for |00>, |01>, |10>, |11>.
        """
        if weights is None:
            weights = self.params
        return self._circuit(inputs, weights)

    def sample(
        self,
        inputs: Tuple[float, float],
        n_samples: int = 1,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Draw samples from the circuit's output distribution.

        Parameters
        ----------
        inputs : tuple[float, float]
            Two classical input values.
        n_samples : int
            Number of measurement samples to draw.
        weights : np.ndarray, optional
            Parameter vector. If None, the current parameters are used.

        Returns
        -------
        samples : np.ndarray
            Array of shape (n_samples, 2) with one‑hot encoded samples.
        """
        probs = self.forward(inputs, weights)
        samples = np.random.choice(len(probs), size=n_samples, p=probs)
        return np.eye(4)[samples]

    def loss_fn(
        self,
        inputs: Tuple[float, float],
        target: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """
        Cross‑entropy loss between circuit output and a target distribution.

        Parameters
        ----------
        inputs : tuple[float, float]
            Classical inputs.
        target : np.ndarray
            One‑hot or probability target of shape (4,).
        weights : np.ndarray, optional
            Parameter vector. If None, current parameters are used.

        Returns
        -------
        loss : float
            Cross‑entropy loss value.
        """
        probs = self.forward(inputs, weights)
        # Add epsilon for numerical stability
        eps = 1e-10
        return -np.sum(target * np.log(probs + eps))

    def fit(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> None:
        """
        Train the variational parameters to match target distributions.

        Parameters
        ----------
        data : np.ndarray
            Input data of shape (N, 2).
        targets : np.ndarray
            Target distributions of shape (N, 4).
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        """
        opt = AdamOptimizer(stepsize=lr)
        for _ in range(epochs):
            loss = 0.0
            for x, y in zip(data, targets):
                loss += self.loss_fn(x, y, self.params)
            loss /= len(data)
            grads = qml.grad(self.loss_fn)(data[0], targets[0], self.params)
            self.params = opt.step(self.params, grads)

__all__ = ["SamplerQNN"]
