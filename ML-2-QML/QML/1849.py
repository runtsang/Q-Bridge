"""
Quantum sampler based on Pennylane.

The implementation uses a variational circuit with:
    - A feature‑map encoding of the classical input.
    - Two layers of entangling CNOTs.
    - Parameterised Ry rotations as trainable weights.
    - Sampling via repeated measurement on a quantum device.

The class exposes a `__call__` method that returns the probability vector
and a `sample` method that returns raw measurement outcomes.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Tuple


class SamplerQNN:
    """
    Quantum sampler neural network.

    Parameters
    ----------
    device : str or pennylane.Device, optional
        Quantum simulator device. Defaults to the default qubit device.
    num_layers : int, optional
        Number of entangling layers. Defaults to 2.
    """

    def __init__(self, device: str | qml.Device | None = None, num_layers: int = 2) -> None:
        self.device = qml.device("default.qubit", wires=2) if device is None else device
        self.num_layers = num_layers
        # Parameter vector: 2 input angles + 4 weight angles per layer
        self.params = np.random.uniform(0, 2 * np.pi, (2 + 4 * self.num_layers), requires_grad=True)

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: Tuple[float, float], params: np.ndarray) -> np.ndarray:
            # Feature‑map: rotate each qubit by input angles
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Parameterised layers
            for l in range(self.num_layers):
                start = 2 + l * 4
                qml.RY(params[start], wires=0)
                qml.RY(params[start + 1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(params[start + 2], wires=0)
                qml.RY(params[start + 3], wires=1)
                qml.CNOT(wires=[0, 1])

            # Measurement in computational basis
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def __call__(self, inputs: Tuple[float, float]) -> np.ndarray:
        """
        Evaluate the sampler for a single input.

        Parameters
        ----------
        inputs : tuple of float
            Classical input vector of length 2.

        Returns
        -------
        np.ndarray
            Probability vector of shape (4,) corresponding to 00, 01, 10, 11.
        """
        return self.circuit(inputs, self.params)

    def sample(self, inputs: Tuple[float, float], num_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the quantum sampler.

        Parameters
        ----------
        inputs : tuple of float
            Classical input vector.
        num_samples : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Array of sampled bitstrings of shape (num_samples, 2).
        """
        probs = self.__call__(inputs)
        return np.random.choice(a=4, size=num_samples, p=probs).reshape(num_samples, 2)

__all__ = ["SamplerQNN"]
