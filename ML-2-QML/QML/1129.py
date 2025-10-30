"""SamplerQNNGen – a variational quantum sampler built with Pennylane.

The circuit now contains two parameterised layers of Ry rotations
followed by a controlled‑Z entangling block, and a final read‑out
measurement that yields a probability distribution over two qubits.
The sampler can be used with Pennylane's `Sampler` backend to
draw samples directly from the quantum state.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import Sampler
from pennylane import qnode
from typing import Tuple


def _quantum_circuit(inputs: Tuple[pnp.ndarray, pnp.ndarray],
                     weights: Tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray, pnp.ndarray]) -> Tuple[pnp.ndarray, pnp.ndarray]:
    """Parameterized circuit that maps two classical inputs and four variational weights to a 2‑qubit state."""
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)

    # First entangling layer
    qml.CZ(wires=[0, 1])

    # Parameterised rotation layer
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)

    # Second entangling layer
    qml.CZ(wires=[0, 1])

    # Final rotation layer
    qml.RY(weights[2], wires=0)
    qml.RY(weights[3], wires=1)

    # Return the statevector for sampling
    return qml.state()


class SamplerQNN:
    """Quantum sampler that wraps a Pennylane QNode and a Sampler backend."""

    def __init__(self, device: str = "default.qubit", shots: int = 1024) -> None:
        self.dev = qml.device(device, wires=2)
        self.shots = shots
        self.sampler = Sampler(self._qnode)

    def _qnode(self, inputs: Tuple[pnp.ndarray, pnp.ndarray],
               weights: Tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray, pnp.ndarray]) -> Tuple[pnp.ndarray, pnp.ndarray]:
        """QNode that returns the statevector for the sampler."""
        return _quantum_circuit(inputs, weights)

    def sample(self, inputs: Tuple[float, float],
               weights: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """Draw a single sample from the quantum circuit."""
        probs = self.sampler(inputs, weights, shots=self.shots)
        # Convert probabilities to a categorical distribution
        outcomes = pnp.array([pnp.array([0, 1]) for _ in range(2)])
        return pnp.random.choice(outcomes, p=probs)

    def get_probabilities(self, inputs: Tuple[float, float],
                          weights: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Return the probability of each basis state."""
        probs = self.sampler(inputs, weights, shots=self.shots)
        return tuple(probs)


def SamplerQNN() -> SamplerQNN:
    """Factory that returns the extended quantum sampler."""
    return SamplerQNN()


__all__ = ["SamplerQNN"]
