"""Hybrid quantum sampler‑estimator network.

This class implements a 3‑qubit variational circuit that produces both
a 2‑class probability distribution (via measurement of qubit 0) and a
regression value (via expectation of Pauli‑Y on qubit 2). The design
merges the sampler and estimator concepts from the two reference pairs
into a single ansatz, enabling joint optimisation of both outputs.
"""

import pennylane as qml
from pennylane import numpy as np
from typing import Dict, Tuple

class SamplerQNN:
    """
    Quantum sampler‑estimator network.

    Attributes
    ----------
    device : qml.Device
        PennyLane quantum device.
    weights : np.ndarray
        Trainable parameters (shape (5,)).
    """

    def __init__(self, device_name: str = "default.qubit", wires: int = 3) -> None:
        self.device = qml.device(device_name, wires=wires)

        # Trainable parameters: 4 sampler weights + 1 estimator weight
        self.weights = np.zeros(5, requires_grad=True)

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Variational ansatz combining sampler and estimator sub‑circuits."""
            # Sampler part: two qubits (0, 1)
            qml.Ry(inputs[0], wires=0)
            qml.Ry(inputs[1], wires=1)
            qml.CNOT(wires=[0, 1])

            qml.Ry(weights[0], wires=0)
            qml.Ry(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])

            qml.Ry(weights[2], wires=0)
            qml.Ry(weights[3], wires=1)

            # Estimator part: one qubit (2)
            qml.Hadamard(wires=2)
            qml.Ry(inputs[2], wires=2)
            qml.RX(weights[4], wires=2)

            # Sampler output: probabilities from qubit 0
            z_exp = qml.expval(qml.PauliZ(0))
            p0 = (1 + z_exp) / 2
            p1 = 1 - p0

            # Estimator output: expectation of Y on qubit 2
            y_exp = qml.expval(qml.PauliY(2))

            return p0, p1, y_exp

        self._circuit = circuit

    def __call__(self, inputs: np.ndarray, weights: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        """
        Evaluate the circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Input vector of shape (3,).
        weights : np.ndarray, optional
            Weight vector of shape (5,). If None, the object's current weights are used.

        Returns
        -------
        output : dict
            {'probabilities': np.ndarray of shape (2,),
             'prediction': np.ndarray of shape (1,)}
        """
        if weights is None:
            weights = self.weights
        p0, p1, y_exp = self._circuit(inputs, weights)
        return {"probabilities": np.array([p0, p1]), "prediction": np.array([y_exp])}

    def sample(self, num_samples: int, inputs: np.ndarray) -> np.ndarray:
        """
        Draw classical samples from the sampler output.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        inputs : np.ndarray
            Input vector for the sampler portion.

        Returns
        -------
        samples : np.ndarray
            Array of shape (num_samples,) with values 0 or 1.
        """
        probs = self.__call__(inputs)["probabilities"]
        return np.random.choice([0, 1], size=num_samples, p=probs)

    def set_weights(self, new_weights: np.ndarray) -> None:
        """
        Replace the current trainable parameters.

        Parameters
        ----------
        new_weights : np.ndarray
            New weight array of shape (5,).
        """
        self.weights = new_weights

    def get_weights(self) -> np.ndarray:
        """
        Retrieve the current weight vector.

        Returns
        -------
        np.ndarray
            Current weights.
        """
        return self.weights

__all__ = ["SamplerQNN"]
