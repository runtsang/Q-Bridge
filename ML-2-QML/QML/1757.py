"""Quantum self‑attention with a parameterised variational circuit.

The circuit implements a learnable rotation and entanglement pattern
mirroring the classical attention parameters.  The output is a
probability distribution over the qubits that can be interpreted
as attention scores.
"""

import pennylane as qml
import numpy as np

class SelfAttention:
    """Variational self‑attention circuit."""
    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_ops()

    def _build_ops(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(rotation_params, entangle_params, inputs):
            # Encode inputs as basis states (simplified)
            for w, val in enumerate(inputs):
                if val > 0.5:
                    qml.PauliX(w)
            # Parameterised rotations
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    idx = (layer * self.n_qubits + qubit) * 3
                    qml.RX(rotation_params[idx], wires=qubit)
                    qml.RY(rotation_params[idx + 1], wires=qubit)
                    qml.RZ(rotation_params[idx + 2], wires=qubit)
                # Entanglement
                for qubit in range(self.n_qubits - 1):
                    gate = entangle_params[qubit]
                    qml.CRX(gate, wires=[qubit, qubit + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened rotation angles (3 * n_qubits * n_layers).
        entangle_params : np.ndarray
            Entanglement angles per qubit (n_qubits - 1).
        inputs : np.ndarray
            Binary input vector of length n_qubits.
        shots : int
            Number of shots for measurement (ignored by default backend).

        Returns
        -------
        np.ndarray
            Normalised attention weights derived from Z‑expectation values.
        """
        expvals = self.circuit(rotation_params, entangle_params, inputs)
        probs = (np.array(expvals) + 1.0) / 2.0
        return probs / probs.sum()

__all__ = ["SelfAttention"]
