"""
Quantum sampler using Pennylane’s strongly entangling layers.

Features
--------
* Parameterised rotation layer per qubit.
* Multiple entangling layers for expressivity.
* Direct probability distribution and sampling via QNode.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import Device


class SamplerQNN:
    """
    Variational quantum sampler.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (default 2).
    device : str or pennylane.Device
        Backend device name or instance.
    layers : int
        Number of entangling layers to stack.
    """

    def __init__(self, num_qubits: int = 2, device: str | Device = "default.qubit", layers: int = 2) -> None:
        self.num_qubits = num_qubits
        self.device = qml.device(device, wires=num_qubits)
        self.layers = layers

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Input rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Entangling layers
            for _ in range(layers):
                qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))

            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def probability_distribution(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the full probability distribution for the given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape ``(num_qubits,)`` with rotation angles.

        Returns
        -------
        np.ndarray
            Probability vector of shape ``(2**num_qubits,)``.
        """
        # Initialise random weights for demonstration; in practice these would be learned.
        rng = np.random.default_rng()
        weights = rng.normal(size=(self.layers, self.num_qubits, 3))
        return self.circuit(inputs, weights)

    def sample(self, inputs: np.ndarray, num_samples: int = 1000, seed: int | None = None) -> np.ndarray:
        """
        Draw samples from the quantum sampler.

        Parameters
        ----------
        inputs : np.ndarray
            Input rotation angles, shape ``(num_qubits,)``.
        num_samples : int
            Number of samples to draw.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Bitstring samples of shape ``(num_samples, num_qubits)``.
        """
        probs = self.probability_distribution(inputs)
        rng = np.random.default_rng(seed)
        return rng.choice(len(probs), size=num_samples, p=probs).reshape(-1, self.num_qubits)

__all__ = ["SamplerQNN"]
