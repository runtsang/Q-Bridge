"""Quantum self‑attention implemented with PennyLane.
The class shares the legacy ``SelfAttention()`` interface but uses a
parameterised quantum circuit to produce attention‑like outputs.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttentionLayer:
    """
    Quantum self‑attention layer.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits representing the embedding dimension.  Defaults to 4.
    device : str or pennylane.Device, optional
        PennyLane device to execute the circuit on.  Defaults to
        ``default.qubit`` simulator.
    shots : int, optional
        Number of measurement shots when using a real device.  Defaults to 1024.
    """

    def __init__(self, n_qubits: int = 4, device=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        if device is None:
            device = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self.device = device
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="autograd")
        def circuit(rotation_params: pnp.ndarray, entangle_params: pnp.ndarray):
            # Apply parameterised rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement pattern: nearest‑neighbour CRX
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Expectation values of Z on each qubit as the “attention” output
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the circuit and return the expectation values.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the single‑qubit rotations.  Shape must be
            (3 * n_qubits,).
        entangle_params : np.ndarray
            Parameters for the CRX gates.  Shape must be (n_qubits - 1,).
        """
        if rotation_params.size!= 3 * self.n_qubits:
            raise ValueError(
                f"rotation_params must have size {3 * self.n_qubits} but got "
                f"{rotation_params.size}"
            )
        if entangle_params.size!= self.n_qubits - 1:
            raise ValueError(
                f"entangle_params must have size {self.n_qubits - 1} but got "
                f"{entangle_params.size}"
            )
        out = self._circuit(rotation_params, entangle_params)
        return np.asarray(out)

    def __repr__(self):
        return f"<SelfAttentionLayer n_qubits={self.n_qubits}>"


def SelfAttention() -> SelfAttentionLayer:
    """
    Factory function compatible with the legacy ``SelfAttention`` module.
    Returns a quantum layer with 4 qubits and the default simulator.
    """
    return SelfAttentionLayer(n_qubits=4, shots=1024)


__all__ = ["SelfAttention", "SelfAttentionLayer"]
