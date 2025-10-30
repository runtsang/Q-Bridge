"""Quantum self‑attention block using Pennylane.

Features
--------
* Multi‑head variational circuits
* Amplitude‑based readout to approximate attention weights
* Optional shot‑based sampling for realistic backends
"""

import pennylane as qml
import numpy as np
from typing import Optional


class SelfAttention:
    """Hybrid variational self‑attention.

    Parameters
    ----------
    n_qubits : int
        Total qubits per head (including 1 qubit for each register of query/key/value).
    num_heads : int, default 1
        Number of attention heads.
    """

    def __init__(self, n_qubits: int, num_heads: int = 1):
        self.n_qubits = n_qubits
        self.num_heads = num_heads
        self.head_qubits = n_qubits // 3  # q, k, v per head

        # Create a device for each head
        self.devices = [qml.Device("default.qubit", wires=self.head_qubits * 3) for _ in range(num_heads)]

    def _attention_circuit(self, seq_len: int):
        """Return a QNode that outputs amplitude‑based attention weights."""
        dev = self.devices[0]  # all heads share the same device layout for simplicity

        @qml.qnode(dev, interface="numpy")
        def circuit(rotation_params, entangle_params):
            # Encode input tokens as computational basis states
            for i in range(seq_len):
                qml.PauliX(i)  # placeholder: in practice encode via state preparation

            # Apply parameterized rotations per head
            for h in range(self.num_heads):
                base = h * self.head_qubits
                for q in range(self.head_qubits):
                    idx = base + q
                    r = rotation_params[h, q]
                    qml.RY(r, idx)
                    qml.RZ(r, idx)

            # Entanglement between heads
            for h in range(self.num_heads - 1):
                qml.CNOT(h * self.head_qubits, (h + 1) * self.head_qubits)

            # Measurement in Z basis to extract probabilities
            return [qml.expval(qml.PauliZ(i)) for i in range(seq_len)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        seq_len: int,
        shots: Optional[int] = None,
    ) -> np.ndarray:
        """
        Execute the variational attention circuit and return probabilities.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (num_heads, head_qubits) – rotational angles for each qubit in each head.
        entangle_params : np.ndarray
            Shape (num_heads,) – entanglement strengths (unused in this toy example).
        seq_len : int
            Number of tokens in the sequence.
        shots : int, optional
            Number of shots for sampling. If None, use statevector.

        Returns
        -------
        np.ndarray
            Shape (seq_len,) – probability distribution approximating attention weights.
        """
        circuit = self._attention_circuit(seq_len)
        if shots is None:
            probs = circuit(rotation_params, entangle_params)
        else:
            probs = circuit(rotation_params, entangle_params, shots=shots)
        return np.array(probs)
__all__ = ["SelfAttention"]
