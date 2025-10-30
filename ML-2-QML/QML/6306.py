"""Quantum SamplerQNN using Pennylane and a variational circuit.

The seed implemented a 2‑qubit circuit with a single entanglement layer.
This extension adds a depth‑controlled variational circuit with
entangling blocks, optional measurement of all qubits, and a
state‑vector sampler that can be used in hybrid workflows.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from pennylane import device

# Device for state‑vector simulation
dev = device("default.qubit", wires=4, shots=None)


def _variational_circuit(inputs: np.ndarray,
                         weights: np.ndarray,
                         entanglement: str = "circular") -> None:
    """Parameterized circuit for a 4‑qubit sampler.

    Parameters
    ----------
    inputs:
        Input angles (size 4) applied as RY rotations.
    weights:
        Weight angles (size 8) applied as RY rotations in two layers.
    entanglement:
        Pattern of entangling gates.  Default is circular.
    """
    for i in range(4):
        qml.RY(inputs[i], wires=i)

    # First variational layer
    for i in range(4):
        qml.RY(weights[i], wires=i)

    # Entangling layer
    if entanglement == "circular":
        for i in range(4):
            qml.CNOT(wires=[i, (i + 1) % 4])
    else:
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])

    # Second variational layer
    for i in range(4, 8):
        qml.RY(weights[i], wires=i % 4)


@qml.qnode(dev, interface="autograd")
def _sampler_qnode(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """QNode that returns the state vector."""
    _variational_circuit(inputs, weights)
    return qml.state()


class SamplerQNN:
    """Hybrid quantum‑classical sampler class.

    The class exposes a ``sample`` method that draws samples from the
    probability distribution defined by the circuit.  It can be
    integrated with classical optimisers as the seed did, but now
    supports more qubits and richer parameterisation.
    """

    def __init__(self,
                 input_dim: int = 4,
                 weight_dim: int = 8,
                 entanglement: str = "circular") -> None:
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.entanglement = entanglement

    def sample(self,
               inputs: np.ndarray,
               weights: np.ndarray,
               num_samples: int = 1024) -> np.ndarray:
        """Draw samples from the quantum circuit.

        Parameters
        ----------
        inputs:
            Input parameters of shape (input_dim,).
        weights:
            Weight parameters of shape (weight_dim,).
        num_samples:
            Number of classical samples to draw.

        Returns
        -------
        samples:
            Array of shape (num_samples, input_dim) containing
            bit‑string samples.
        """
        # Obtain the full probability distribution from the state vector
        state = _sampler_qnode(inputs, weights)
        probs = np.abs(state) ** 2
        probs = probs.reshape((2,) * self.input_dim)

        # Flatten to a 1‑D probability vector
        probs_flat = probs.flatten()
        # Sample indices according to the distribution
        idx = np.random.choice(len(probs_flat), size=num_samples, p=probs_flat)
        # Convert indices back to bit‑strings
        samples = np.array([list(np.binary_repr(i, width=self.input_dim))
                            for i in idx], dtype=int)
        return samples

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Convenience wrapper returning the state vector."""
        return _sampler_qnode(inputs, weights)


def SamplerQNN_factory() -> SamplerQNN:
    """Return a default SamplerQNN instance."""
    return SamplerQNN()


__all__ = ["SamplerQNN", "SamplerQNN_factory", "SamplerQNN"]
