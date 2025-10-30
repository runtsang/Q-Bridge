"""Quantum sampler network built on Pennylane with entangling layers and sampling support."""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Optional, Sequence

class SamplerQNN:
    """
    Variational sampler implemented as a parameterised quantum circuit.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input parameters (one RY rotation per wire).
    n_qubits : int, default 2
        Number of qubits / wires in the circuit.
    n_layers : int, default 2
        Depth of the variational block.
    device : str, default "default.qubit"
        Backend device name.
    """
    def __init__(
        self,
        input_dim: int = 2,
        n_qubits: int = 2,
        n_layers: int = 2,
        device: str = "default.qubit",
    ) -> None:
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self._build_qnode()
        # Initialise trainable weights uniformly in [-π, π]
        self.params = pnp.random.uniform(-np.pi, np.pi, self.qnode.n_params)

    def _build_qnode(self) -> None:
        dev = qml.device(self.device, wires=self.n_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(inputs, weights):
            # Input encoding
            for i, inp in enumerate(inputs):
                qml.RY(inp, wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RY(weights[layer, qubit], wires=qubit)
                # Ring‑type entanglement
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Return probability distribution over all computational basis states
            return qml.probs(wires=range(self.n_qubits))

        self.qnode = circuit

    def forward(self, inputs: Sequence[float], params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate the circuit and return a probability vector.

        Parameters
        ----------
        inputs : Sequence[float]
            Input angles for the RY encoding.
        params : np.ndarray, optional
            Variational parameters; if None the stored parameters are used.

        Returns
        -------
        np.ndarray
            Probability vector of shape (2**n_qubits,).
        """
        if params is None:
            params = self.params
        return self.qnode(np.asarray(inputs), np.asarray(params))

    def sample(
        self,
        inputs: Sequence[float],
        params: Optional[np.ndarray] = None,
        num_samples: int = 1,
    ) -> np.ndarray:
        """
        Draw samples from the quantum device output distribution.

        Parameters
        ----------
        inputs : Sequence[float]
            Input angles for the RY encoding.
        params : np.ndarray, optional
            Variational parameters; if None the stored parameters are used.
        num_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array of sampled basis‑state indices.
        """
        probs = self.forward(inputs, params)
        return np.random.choice(len(probs), size=num_samples, p=probs)

    def get_params(self) -> np.ndarray:
        """Return the current trainable parameter array."""
        return self.params

    def set_params(self, new_params: np.ndarray) -> None:
        """Replace the current parameter array with `new_params`."""
        self.params = new_params

    def trainable_params(self):
        """Return a list of trainable parameters for optimisation."""
        return list(self.qnode.trainable_params)

def SamplerQNN() -> SamplerQNN:
    """
    Factory that returns a ready‑to‑train quantum sampler.

    The returned instance encapsulates a 2‑qubit variational circuit with a
    two‑layer entangling block, expanding the original 1‑layer example
    while keeping the same public API.
    """
    return SamplerQNN()

__all__ = ["SamplerQNN"]
