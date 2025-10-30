"""
EnhancedSamplerQNN: A variational quantum sampler.

Features
--------
* Two‑qubit parameterised circuit with configurable entangling layers.
* Uses Pennylane's default qubit simulator.
* Provides probability distribution via expectation values of Pauli‑Z.
* Sampling method draws from the quantum probability distribution.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Tuple


class EnhancedSamplerQNN:
    """
    Variational quantum sampler based on a two‑qubit circuit.

    Parameters
    ----------
    dev : str or qml.Device, default "default.qubit"
        Quantum device or device name.
    wires : int, default 2
        Number of qubits.
    n_layers : int, default 2
        Number of entangling layers.
    seed : int, default 0
        Random seed for weight initialization.
    """

    def __init__(
        self,
        dev: str | qml.Device = "default.qubit",
        wires: int = 2,
        n_layers: int = 2,
        seed: int = 0,
    ) -> None:
        self.dev = dev if isinstance(dev, qml.Device) else qml.device(dev, wires=wires)
        self.wires = wires
        self.n_layers = n_layers
        np.random.seed(seed)
        # Weight shape: (n_layers, wires)
        self.weights = pnp.random.uniform(0, 2 * np.pi, size=(n_layers, wires))
        self._build_circuit()

    def _entangle_layer(self, layer: int) -> None:
        """Apply an entangling layer of CNOTs."""
        for i in range(self.wires - 1):
            qml.CNOT(wires=[i, i + 1])
        # Wrap around for cyclic entanglement
        qml.CNOT(wires=[self.wires - 1, 0])

    def _circuit(self, inputs: Tuple[pnp.ndarray,...]) -> None:
        """Parameterized circuit that accepts both input and weight parameters."""
        # Input rotations
        for i, inp in enumerate(inputs):
            qml.RY(inp, wires=i)
        # Variational layers
        for layer in range(self.n_layers):
            qml.RY(self.weights[layer, 0], wires=0)
            qml.RY(self.weights[layer, 1], wires=1)
            self._entangle_layer(layer)

    @qml.qnode
    def _qnode(self, inputs: Tuple[pnp.ndarray,...]) -> Tuple[pnp.ndarray,...]:
        """QNode that returns the probability of measuring |00> and |11>."""
        self._circuit(inputs)
        # Return probabilities for computational basis states
        return qml.probs(wires=self.wires)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution for given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (..., wires) containing input angles in radians.

        Returns
        -------
        np.ndarray
            Probability vector of shape (..., 4) corresponding to |00>, |01>, |10>, |11>.
        """
        probs = self._qnode(tuple(inputs.T))
        return probs

    def sample(self, inputs: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the quantum probability distribution.

        Parameters
        ----------
        inputs : np.ndarray
            Input angles.
        n_samples : int, default 1
            Number of samples per input.

        Returns
        -------
        np.ndarray
            Sample indices of shape (n_samples, *inputs.shape[:-1]).
        """
        probs = self.__call__(inputs)
        flat_probs = probs.reshape(-1, probs.shape[-1])
        samples = []
        for prob in flat_probs:
            samples.append(np.random.choice(4, size=n_samples, p=prob))
        return np.array(samples).reshape(n_samples, *inputs.shape[:-1])

__all__ = ["EnhancedSamplerQNN"]
