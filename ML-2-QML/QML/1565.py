"""
Quantum SamplerQNN implementation using Pennylane.

Features:
- Variational circuit with alternating rotation and entanglement layers.
- Parameter‑shift gradients available via Pennylane QNode.
- Supports sampling via a QNode returning probability amplitudes.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Sequence, Iterable


class SamplerQNN:
    """
    Quantum sampler based on a parameterised variational circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the sampler.
    n_layers : int
        Number of alternating rotation / entanglement layers.
    entanglement : str | Sequence[tuple[int, int]]
        Entanglement scheme. Defaults to "circular".
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 3,
        entanglement: str | Sequence[tuple[int, int]] = "circular",
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Device for simulation; can be swapped for real hardware.
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameter shape: (n_layers, n_qubits)
        self.n_params = n_layers * n_qubits

        # Initialise all parameters to zero; they will be trained.
        self.params = np.zeros(self.n_params, requires_grad=True)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(p: Iterable[float]) -> qml.Result:
            """Variational circuit with rotation and entanglement layers."""
            # Unpack parameters
            p = np.reshape(p, (self.n_layers, self.n_qubits))
            for layer in range(self.n_layers):
                # Rotation layer
                for q in range(self.n_qubits):
                    qml.RY(p[layer, q], wires=q)
                # Entanglement layer
                if entanglement == "circular":
                    for q in range(self.n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
                else:
                    for (q1, q2) in entanglement:
                        qml.CNOT(wires=[q1, q2])
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def forward(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Return probability distribution over all possible bit‑strings.

        Parameters
        ----------
        params : array-like | None
            Optional parameter override. If None, uses self.params.

        Returns
        -------
        np.ndarray
            Probabilities of shape (2**n_qubits,).
        """
        p = params if params is not None else self.params
        return self.circuit(p)

    def sample(self, n_shots: int = 1024) -> np.ndarray:
        """
        Draw samples from the circuit.

        Parameters
        ----------
        n_shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Array of bit‑string samples (shape: (n_shots, n_qubits)).
        """
        probs = self.forward()
        outcomes = np.arange(2 ** self.n_qubits)
        samples = np.random.choice(outcomes, size=n_shots, p=probs)
        # Convert integer samples to binary strings
        bits = ((samples[:, None] & (1 << np.arange(self.n_qubits)[::-1])) > 0).astype(int)
        return bits

    def get_params(self) -> np.ndarray:
        """Return current parameters."""
        return self.params

    def set_params(self, new_params: np.ndarray) -> None:
        """Set new parameters."""
        self.params = new_params

__all__ = ["SamplerQNN"]
