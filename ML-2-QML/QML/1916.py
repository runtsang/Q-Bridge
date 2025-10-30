"""Variational quantum self‑attention circuit using Pennylane.

The circuit encodes the input as rotation angles, applies a depth‑controlled
parameterised ansatz, and measures Pauli‑Z expectation values.  These values
are post‑processed with a softmax to obtain a probability distribution that
mimics attention weights.
"""

import numpy as np
import pennylane as qml

class VariationalSelfAttention:
    """Quantum self‑attention block.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int, default 2
        Depth of the parameterised ansatz.
    """
    def __init__(self, num_qubits: int, depth: int = 2):
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev)
        def circuit(params: np.ndarray, inputs: np.ndarray):
            # Encode classical inputs as Y‑rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Parameterised rotation ansatz
            idx = 0
            for _ in range(depth):
                for i in range(num_qubits):
                    qml.RX(params[idx], wires=i); idx += 1
                    qml.RY(params[idx], wires=i); idx += 1
                    qml.RZ(params[idx], wires=i); idx += 1
                # Entangle neighbouring qubits
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Measure expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024):
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the RX/RX/RZ rotations (length = num_qubits * 3 * depth).
        entangle_params : np.ndarray
            Unused in the current ansatz but kept for API compatibility.
        inputs : np.ndarray
            Input angles for the RY encoding (length = num_qubits).
        shots : int, default 1024
            Number of shots (ignored on the default simulator).

        Returns
        -------
        np.ndarray
            Softmax‑normalised probability vector of shape (num_qubits,).
        """
        # Concatenate all trainable parameters (entangle_params are ignored but
        # retained to match the original signature).
        params = np.concatenate([rotation_params, entangle_params])
        # Forward pass
        evals = self.circuit(params, inputs)
        # Convert expectation values to a probability distribution
        logits = np.array(evals)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

def SelfAttention(num_qubits: int = 4, depth: int = 2):
    """Factory that returns a variational quantum self‑attention instance."""
    return VariationalSelfAttention(num_qubits, depth)

__all__ = ["SelfAttention"]
