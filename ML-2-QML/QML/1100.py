"""Quantum self‑attention using a variational circuit with Pennylane.

The circuit implements a parameterized block that mimics the
attention mechanism by encoding the input embeddings as angles
and performing entangling layers.  The output is a set of
measurement counts that can be post‑processed to obtain
attention‑like scores.
"""

import pennylane as qml
import numpy as np

class SelfAttentionEnhanced:
    """Variational quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used to encode the input sequence.
    num_layers : int, optional
        Depth of the variational ansatz.
    """
    def __init__(self, n_qubits: int, num_layers: int = 2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameter shapes
        self.rotation_shape = (n_qubits, 3)  # RX, RY, RZ per qubit
        self.entangle_shape = (num_layers, n_qubits - 1)  # CX between neighbors

    def _ansatz(self, rotation_params, entangle_params):
        """Variational ansatz with alternating rotation and entanglement layers."""
        for layer in range(self.num_layers):
            # Rotation layer
            for q in range(self.n_qubits):
                qml.RX(rotation_params[layer, q, 0], wires=q)
                qml.RY(rotation_params[layer, q, 1], wires=q)
                qml.RZ(rotation_params[layer, q, 2], wires=q)
            # Entanglement layer (nearest‑neighbour CX)
            for q in range(self.n_qubits - 1):
                qml.CX(wires=[q, q + 1])

    def run(
        self,
        backend,  # kept for API compatibility; ignored in this implementation
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the variational self‑attention circuit.

        Parameters
        ----------
        backend : Any
            Ignored; placeholder for API compatibility.
        rotation_params : np.ndarray
            Parameters for the rotation gates. Shape
            (num_layers, n_qubits, 3).
        entangle_params : np.ndarray
            Parameters for the entanglement gates (unused in the ansatz
            but kept for API compatibility).
        shots : int, optional
            Number of measurement shots.

        Returns
        -------
        dict
            Measurement counts mapping bitstring to frequency.
        """
        # Reshape parameters to match expected shapes
        rot = rotation_params.reshape(self.num_layers, self.n_qubits, 3)
        ent = entangle_params.reshape(self.num_layers, self.n_qubits - 1)

        @qml.qnode(self.dev, interface="numpy")
        def circuit():
            self._ansatz(rot, ent)
            return qml.probs(wires=range(self.n_qubits))

        probs = circuit()
        # Scale to requested shots
        counts = {format(i, f"0{self.n_qubits}b"): int(p * shots) for i, p in enumerate(probs)}
        return counts

__all__ = ["SelfAttentionEnhanced"]
