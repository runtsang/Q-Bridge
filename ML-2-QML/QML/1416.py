"""Quantum self‑attention using PennyLane.

The circuit encodes the input tokens as rotation angles,
applies a parameterised rotation layer and controlled
entanglement, and measures the expectation of Pauli‑Z
on each qubit.  The resulting marginal probabilities
serve as attention weights, mirroring the classical
interface.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttentionModel:
    """
    Quantum self‑attention model.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (tokens) per input sample.
    dev_name : str, default "default.qubit"
        PennyLane device name.
    """

    def __init__(self, n_qubits: int, dev_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)

    def _attention_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray, sample: np.ndarray
    ):
        """
        Internal circuit building block.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3) – RX, RY, RZ angles for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits - 1,) – rotation angles for controlled‑RX gates.
        sample : np.ndarray
            Shape (n_qubits,) – input token values.
        """
        for i in range(self.n_qubits):
            # Encode classical input as RX rotation
            qml.RX(sample[i], wires=i)

        for i in range(self.n_qubits):
            qml.RX(rotation_params[i, 0], wires=i)
            qml.RY(rotation_params[i, 1], wires=i)
            qml.RZ(rotation_params[i, 2], wires=i)

        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit for each sample in the batch and return
        marginal probabilities of each qubit being |1⟩.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3).
        entangle_params : np.ndarray
            Shape (n_qubits - 1,).
        inputs : np.ndarray
            Shape (batch, n_qubits).
        shots : int, default 1024
            Number of shots for the simulator.

        Returns
        -------
        np.ndarray
            Attention scores of shape (batch, n_qubits).
        """
        batch_size = inputs.shape[0]
        attn_scores = np.zeros((batch_size, self.n_qubits), dtype=np.float32)

        @qml.qnode(self.dev, interface="autograd", shots=shots)
        def qnode(sample):
            return self._attention_circuit(rotation_params, entangle_params, sample)

        for idx in range(batch_size):
            probs = qnode(inputs[idx])
            # Convert expectation values to probabilities
            probs = (1 - probs) / 2  # PauliZ expectation -> probability of |1⟩
            attn_scores[idx] = probs

        return attn_scores


__all__ = ["SelfAttentionModel"]
