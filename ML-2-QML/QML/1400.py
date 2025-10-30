"""Quantum sampler using Pennylane with a parameterised entangling circuit."""
from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class SamplerQNNExtended:
    """
    A variational quantum sampler on two qubits.

    The circuit consists of:
    - Two input rotations (ry) per qubit.
    - One entangling CX gate.
    - Two layers of trainable rotations on each qubit.
    - A final entangling CX gate.

    The sampler returns the probability of measuring the |00⟩ state, which
    can be interpreted as a probability distribution over two outcomes
    (|00⟩ vs. all other states).

    Parameters
    ----------
    device_name : str, default "default.qubit"
        Pennylane device backend.
    wires : int, default 2
        Number of qubits.
    """

    def __init__(self, device_name: str = "default.qubit", wires: int = 2) -> None:
        self.device = qml.device(device_name, wires=wires)
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Input rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # First entangling layer
            qml.CNOT(wires=[0, 1])

            # Trainable rotations
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)

            # Probabilities of all basis states
            return qml.probs(wires=[0, 1])

        self.qnode = circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit and return a two‑element probability vector.

        Parameters
        ----------
        inputs : np.ndarray
            Input angles for the two qubits (shape (2,)).
        weights : np.ndarray
            Trainable rotation angles (shape (4,)).

        Returns
        -------
        np.ndarray
            Probabilities [P(|00⟩), P(not |00⟩)].
        """
        probs = self.qnode(inputs, weights)
        return np.array([probs[0], 1 - probs[0]])

    def sample(self, inputs: np.ndarray, weights: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Draw samples from the circuit using the Pennylane simulator.

        Parameters
        ----------
        inputs : np.ndarray
            Input angles for the two qubits.
        weights : np.ndarray
            Trainable rotation angles.
        shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Sampled counts for the two outcomes.
        """
        self.device = qml.device("default.qubit", wires=2, shots=shots)
        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            return qml.probs(wires=[0, 1])

        probs = circuit(inputs, weights)
        return np.round(probs * shots).astype(int)

def SamplerQNNExtended() -> SamplerQNNExtended:
    """Return an instance of the enhanced quantum sampler."""
    return SamplerQNNExtended()

__all__ = ["SamplerQNNExtended"]
