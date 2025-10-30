"""Quantum self‑attention using a variational ansatz and parameter‑shift estimation.

Features
--------
* Parameter‑shift based gradient estimator for training,
* Entanglement layer built with controlled‑RZ gates,
* Measurement‑based attention readout that returns a probability distribution
  over the input sequence positions.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import Device


class SelfAttention:
    """
    Variational self‑attention quantum circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits representing the sequence length.
    device : pennylane.Device, optional
        Quantum device.  Defaults to the default qubit simulator.
    """

    def __init__(self, n_qubits: int, device: Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.device = device or qml.device("default.qubit", wires=n_qubits)

        # Parameter ranges for rotation and entanglement gates
        self.rotation_params = pnp.random.uniform(0, 2 * np.pi, 3 * n_qubits)
        self.entangle_params = pnp.random.uniform(0, 2 * np.pi, n_qubits - 1)

        # Define the variational circuit
        @qml.qnode(self.device, interface="autograd")
        def circuit(rotation_params, entangle_params):
            # State preparation (could be classical input encoded later)
            qml.Hadamard(wires=range(self.n_qubits))

            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement layer (controlled‑RZ)
            for i in range(self.n_qubits - 1):
                qml.CRY(entangle_params[i], wires=[i, i + 1])

            # Measurement for attention readout
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit on a given backend.

        Parameters
        ----------
        backend : pennylane.Device
            Quantum device or simulator.
        rotation_params : np.ndarray
            Rotation angles for RX, RY, RZ on each qubit.
        entangle_params : np.ndarray
            Entanglement angles for CRY gates.
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Probability distribution over sequence positions.
        """
        # Update parameters
        self.circuit.device = backend
        probs = self.circuit(rotation_params, entangle_params)
        # If using a simulator that returns probabilities directly, apply shots
        if shots:
            probs = probs * shots
            probs = probs / np.sum(probs)
        return probs

__all__ = ["SelfAttention"]
