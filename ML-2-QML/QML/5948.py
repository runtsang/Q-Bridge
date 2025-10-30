"""
Pennylane implementation of a variational sampler network.
The circuit uses two qubits, a parameter‑shift gradient, and
provides a `sample` method that returns a probability distribution
over the computational basis.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Iterable, Tuple


class SamplerQNN:
    """
    Variational sampler built with Pennylane.

    Parameters
    ----------
    wires : int, default 2
        Number of qubits in the circuit.
    shots : int, default 1024
        Number of samples to draw during evaluation.
    device : str, optional
        Pennylane device name (e.g., 'default.qubit', 'qiskit.ibmq.qasm_simulator').
    """

    def __init__(self, wires: int = 2, shots: int = 1024, device: str = "default.qubit") -> None:
        self.wires = wires
        self.shots = shots
        self.dev = qml.device(device, wires=wires, shots=shots)
        self.params = np.random.uniform(0, 2 * np.pi, (4,))  # initial weights

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Tuple[float, float], weights: Iterable[float]) -> Tuple[float, float]:
            # Input encoding
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            # Variational layer
            for w in weights:
                qml.RY(w, wires=0)
                qml.RY(w, wires=1)
                qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        self.circuit = circuit

    def sample(self, inputs: Tuple[float, float]) -> np.ndarray:
        """
        Draw samples from the circuit and return a probability distribution.

        Parameters
        ----------
        inputs : Tuple[float, float]
            Two real numbers that are encoded as RY rotations.

        Returns
        -------
        np.ndarray
            Probability vector of shape (2,) over basis states |00> and |01>.
        """
        probs = self.circuit(inputs, self.params)
        # Convert expectation values to probabilities
        probs = (np.array(probs) + 1) / 2
        return probs

    def loss(self, inputs: Tuple[float, float], target: np.ndarray) -> float:
        """
        Binary cross‑entropy loss between the sampled distribution and a target.

        Parameters
        ----------
        inputs : Tuple[float, float]
            Input parameters for the circuit.
        target : np.ndarray
            One‑hot target vector of shape (2,).

        Returns
        -------
        float
            Loss value.
        """
        probs = self.sample(inputs)
        return -np.sum(target * np.log(probs + 1e-9))

    def train_step(self, inputs: Tuple[float, float], target: np.ndarray, lr: float = 0.01) -> None:
        """
        Perform a single gradient‑descent update on the circuit parameters.

        Parameters
        ----------
        inputs : Tuple[float, float]
            Input parameters.
        target : np.ndarray
            Target distribution.
        lr : float, optional
            Learning rate.
        """
        grad = qml.grad(self.loss, argnums=2)(inputs, target, self.params)
        self.params -= lr * grad

__all__ = ["SamplerQNN"]
