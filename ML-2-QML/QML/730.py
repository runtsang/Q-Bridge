"""
SamplerQNN__gen406 – Quantum variational sampler using PennyLane.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import Iterable, Tuple

__all__ = ["SamplerQNN"]


class SamplerQNN:
    """
    A quantum sampler that implements a parameterized circuit
    and returns samples from the resulting probability distribution.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    rotation : str
        Rotation gate name supported by PennyLane (e.g., "ry", "rx", "rz").
    entanglement : str
        Entanglement pattern: "cnot" (chain) or "full" (all‑to‑all).
    wires : Iterable[int] | None
        Wire indices; defaults to consecutive integers.
    shots : int
        Number of shots for sampling.
    device_name : str
        PennyLane backend device to use.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        rotation: str = "ry",
        entanglement: str = "cnot",
        wires: Iterable[int] | None = None,
        shots: int = 1024,
        device_name: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.rotation = rotation
        self.entanglement = entanglement
        self.wires = list(wires) if wires is not None else list(range(n_qubits))
        self.shots = shots
        self.device = qml.device(device_name, wires=self.wires, shots=self.shots)

        # Initialize weight parameters randomly.
        self.params = np.random.uniform(0, 2 * np.pi, size=(n_qubits * 2,))

        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the variational circuit as a QNode."""

        @qml.qnode(self.device)
        def circuit(*weights: np.ndarray) -> np.ndarray:
            # Parameterised single‑qubit rotations.
            for i, w in enumerate(weights):
                getattr(qml, self.rotation)(w, wires=self.wires[i])

            # Entanglement layer.
            if self.entanglement == "cnot":
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
            elif self.entanglement == "full":
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[self.wires[i], self.wires[j]])

            # Return the joint probability distribution over all qubits.
            return qml.probs(wires=self.wires)

        self.circuit = circuit

    def sample(self, n_samples: int = 1024) -> np.ndarray:
        """
        Draw samples from the quantum circuit's output distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, n_qubits)`` containing binary bitstrings.
        """
        probs = self.circuit(*self.params)
        # Convert probabilities to a cumulative distribution for sampling.
        cumulative = np.cumsum(probs)
        rand_vals = np.random.rand(n_samples)
        indices = np.searchsorted(cumulative, rand_vals)
        # Convert flat indices to bitstrings.
        samples = np.array(
            [
                [(indices[i] >> j) & 1 for j in reversed(range(self.n_qubits))]
                for i in range(n_samples)
            ]
        )
        return samples

    def set_params(self, params: np.ndarray) -> None:
        """
        Replace the circuit's weight parameters.

        Parameters
        ----------
        params : np.ndarray
            Array of shape ``(n_qubits * 2,)``.
        """
        if params.shape!= (self.n_qubits * 2,):
            raise ValueError("Parameter array has incorrect shape.")
        self.params = params

    def get_probs(self) -> np.ndarray:
        """Return the probability vector over all basis states."""
        return self.circuit(*self.params)

    def __repr__(self) -> str:
        return (
            f"SamplerQNN(n_qubits={self.n_qubits}, rotation='{self.rotation}', "
            f"entanglement='{self.entanglement}', shots={self.shots})"
        )
