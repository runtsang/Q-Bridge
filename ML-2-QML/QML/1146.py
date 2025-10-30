"""Quantum sampler network using Pennylane."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from typing import Tuple


class SamplerQNNGen:
    """
    Two‑qubit variational sampler implemented with Pennylane.

    Circuit:
      - Data‑dependent Ry rotations on each qubit.
      - Two layers of CNOT entanglement.
      - Three trainable Ry layers (two per qubit).
    Methods:
      - ``forward`` returns the probability distribution over the 4 computational basis states.
      - ``sample`` draws samples using the simulator; the number of shots can be set per call.
      - ``set_weights`` / ``get_weights`` manage the trainable parameters.
    """

    def __init__(self, device: str = "default.qubit", shots: int = 8192) -> None:
        self.device = qml.device(device, wires=2, shots=shots)
        self.weights = np.random.uniform(0, 2 * np.pi, 4)

        def circuit(inputs: np.ndarray, weights: np.ndarray) -> None:
            for i in range(2):
                qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)

        self._circuit = circuit
        self._probs_qnode = qml.QNode(self._probs, self.device, interface="autograd")
        self._sample_qnode = qml.QNode(self._sample, self.device, interface="autograd")

    def _probs(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._circuit(inputs, weights)
        return qml.probs(wires=[0, 1])

    def _sample(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._circuit(inputs, weights)
        return qml.sample(wires=[0, 1])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Return the probability distribution over the 4 computational basis states.
        """
        return self._probs_qnode(inputs, self.weights)

    def sample(self, inputs: np.ndarray, num_samples: int = 8192) -> np.ndarray:
        """
        Draw samples from the circuit using the simulator.
        """
        self.device.shots = num_samples
        return self._sample_qnode(inputs, self.weights)

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Update trainable parameters.
        """
        self.weights = weights

    def get_weights(self) -> np.ndarray:
        """
        Return current trainable parameters.
        """
        return self.weights
