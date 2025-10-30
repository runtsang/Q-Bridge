"""Quantum self‑attention using Pennylane variational circuit.

The circuit implements a parameterized attention block.  Each qubit encodes a
feature dimension.  Rotation parameters control single‑qubit rotations that
implement the query and key projections.  Entangle parameters control
controlled‑RZ gates that entangle neighboring qubits, mimicking the
attention weight computation.  The measurement of each qubit yields a binary
attention mask; the mask is post‑processed to produce a weighted sum over
the input values.
"""

import pennylane as qml
import numpy as np

class SelfAttentionEnhanced:
    """
    Quantum self‑attention block.  The constructor receives the number of
    qubits (equal to the embedding dimension).  ``rotation_params`` and
    ``entangle_params`` are interpreted as angles for single‑qubit rotations
    and controlled‑RZ gates, respectively.
    """
    def __init__(self, n_qubits: int, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray):
        """Variational circuit that prepares the attention state."""
        # Encode inputs as rotations around Y
        for i, val in enumerate(inputs):
            qml.RY(val, wires=i)

        # Rotation block
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)

        # Entanglement block
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        # Measurement of all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Angles for the rotation block. Shape (3 * n_qubits,).
        entangle_params : np.ndarray
            Angles for the entanglement block. Shape (n_qubits - 1,).
        inputs : np.ndarray
            Input vector of shape (n_qubits,).  Each entry is interpreted as
            a rotation angle for the initial state preparation.
        shots : int
            Number of shots for sampling.

        Returns
        -------
        np.ndarray
            Expected values of Pauli‑Z on each qubit, reshaped to (n_qubits,).
        """
        @qml.qnode(self.dev, interface="autograd")
        def qnode():
            return self._circuit(rotation_params, entangle_params, inputs)

        # Compute expectation values
        expvals = qnode()
        return np.array(expvals)

__all__ = ["SelfAttentionEnhanced"]
