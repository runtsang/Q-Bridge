"""Quantum sampler network using PennyLane variational circuit.

The circuit operates on two qubits and consists of a configurable number of
entanglement layers.  It returns the full probability distribution over the
four computational basis states, enabling sampling with the built‑in
`qml.sample` primitive.  The class exposes a `sample` method that draws
samples directly from the simulator, mirroring the API of the classical
counterpart.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import QNode
from typing import Tuple, Sequence

class SamplerQNNGen163:
    """Variational two‑qubit sampler with entanglement layers."""

    def __init__(self, device_name: str = "default.qubit", n_layers: int = 2,
                 seed: int | None = None) -> None:
        self.dev = qml.device(device_name, wires=2, shots=8192)
        self.n_layers = n_layers
        self.seed = seed

        # Parameter shapes
        self.param_shapes = {
            "rot": (n_layers, 2),
            "ent": (n_layers, 2),
        }

        # Initialise parameters
        self.params = self._init_params()

        # Build QNode
        self.qnode = self._build_qnode()

    def _init_params(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        return {
            "rot": rng.uniform(0, 2 * np.pi, self.param_shapes["rot"]),
            "ent": rng.uniform(0, 2 * np.pi, self.param_shapes["ent"]),
        }

    def _build_qnode(self) -> QNode:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
            # Encode inputs as rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Variational layers
            for layer in range(self.n_layers):
                qml.RY(params["rot"][layer, 0], wires=0)
                qml.RY(params["rot"][layer, 1], wires=1)
                qml.CNOT(0, 1)
                qml.RZ(params["ent"][layer, 0], wires=0)
                qml.RZ(params["ent"][layer, 1], wires=1)

            # Return probabilities of all 4 basis states
            return qml.probs(wires=[0, 1])

        return circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return probability distribution for given 2‑dimensional input."""
        return self.qnode(inputs, self.params)

    def sample(self, inputs: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Draw samples from the quantum sampler."""
        probs = self.forward(inputs)
        # Convert probabilities to a list of basis states
        states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        # Sample indices according to probs
        indices = np.random.choice(len(states), size=n_samples, p=probs)
        return states[indices]

__all__ = ["SamplerQNNGen163"]
