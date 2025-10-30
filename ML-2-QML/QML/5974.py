"""
Quantum sampler based on a variational circuit with two qubits.
The circuit receives two input angles and four trainable weights, and
returns the probability distribution over the computational basis.
"""
from __future__ import annotations

import pennylane as qml
import numpy as np

class SamplerQNN:
    """
    Variational quantum sampler that maps a 2‑dimensional input vector
    and 4 trainable parameters to a 4‑dimensional probability vector
    over the 2‑qubit computational basis.
    """
    def __init__(self,
                 dev: qml.Device | None = None,
                 input_dim: int = 2,
                 weight_dim: int = 4) -> None:
        # Use a default qubit device with 2 wires if none provided
        self.dev = dev or qml.device("default.qubit", wires=2)
        self.input_dim = input_dim
        self.weight_dim = weight_dim

        # Parameter placeholders – these will be passed to the QNode
        self.inputs = np.zeros(input_dim, dtype=np.float64)
        self.weights = np.zeros(weight_dim, dtype=np.float64)

        # Define the variational circuit as a QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Encode input angles
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)
            # Entanglement block
            qml.CNOT(0, 1)
            # Trainable rotations
            qml.RY(weights[0], wires=0)
            qml.RZ(weights[1], wires=1)
            qml.CNOT(0, 1)
            qml.RZ(weights[2], wires=0)
            qml.RX(weights[3], wires=1)
            # Return measurement probabilities
            return qml.probs(wires=[0, 1])

        self._circuit = circuit

    def set_parameters(self, inputs: np.ndarray, weights: np.ndarray) -> None:
        """
        Update the input and weight parameters for the sampler.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (2,) containing the input angles.
        weights : np.ndarray
            Array of shape (4,) containing the trainable weights.
        """
        self.inputs = inputs
        self.weights = weights

    def forward(self) -> np.ndarray:
        """
        Execute the variational circuit and return the probability distribution.

        Returns
        -------
        np.ndarray
            Array of shape (4,) with probabilities for |00>, |01>, |10>, |11>.
        """
        return self._circuit(self.inputs, self.weights)

    def sample(self, num_shots: int = 1024) -> np.ndarray:
        """
        Sample bitstrings from the circuit using the device's sampling method.

        Parameters
        ----------
        num_shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Array of shape (num_shots, 2) containing sampled bitstrings.
        """
        return self.dev.execute(
            self._circuit, shots=num_shots, inputs=self.inputs, weights=self.weights
        )

__all__ = ["SamplerQNN"]
