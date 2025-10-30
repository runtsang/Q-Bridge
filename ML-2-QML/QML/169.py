"""Quantum self‑attention using Pennylane.

The circuit implements a variational block that mirrors the classical
attention flow: rotation gates encode the query/key matrices, CRX
entanglement mimics the key‑query interaction, and measurement
produces a probability distribution that can be interpreted as
attention scores.  The interface matches the classical version
(`run(rotation_params, entangle_params, shots)`), providing a
back‑end‑agnostic execution path.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import device
from typing import Dict, Any


class SelfAttentionModule:
    """
    Variational quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits; must match the dimensionality of the input
        embeddings and the shape of ``rotation_params``.
    num_layers : int, default 1
        Number of stacked rotation‑entanglement layers.
    """

    def __init__(self, n_qubits: int, num_layers: int = 1):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Create a parameter‑ized circuit with rotation and entangling gates."""
        @qml.qnode(self.dev)
        def circuit():
            # Apply rotation gates per qubit
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entangle adjacent qubits with CRX gates
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Repeat layers if requested
            for _ in range(self.num_layers - 1):
                for i in range(self.n_qubits):
                    qml.RX(rotation_params[3 * i], wires=i)
                    qml.RY(rotation_params[3 * i + 1], wires=i)
                    qml.RZ(rotation_params[3 * i + 2], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Measure expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
        backend: str = "default.qubit",
    ) -> Dict[str, Any]:
        """
        Execute the quantum attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (3 * n_qubits,) encoding RX, RY, RZ angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) encoding CRX angles.
        shots : int, default 1024
            Number of circuit executions for sampling.
        backend : str, default 'default.qubit'
            Pennylane device name; can be any supported backend.

        Returns
        -------
        dict
            Dictionary containing expectation values per qubit and the
            raw measurement results (if ``shots`` > 0).
        """
        # Replace device if a different backend is requested
        if backend!= self.dev.name:
            self.dev = device(backend, wires=self.n_qubits)

        circuit = self._circuit(rotation_params, entangle_params)
        # Execute with specified shots
        results = circuit()
        output = {"expectations": results}

        if shots > 0:
            # Perform sampling to obtain a probability distribution
            samples = qml.sample(circuit, shots=shots)
            output["samples"] = samples

        return output


__all__ = ["SelfAttentionModule"]
