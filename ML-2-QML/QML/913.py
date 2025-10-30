"""Quantum self‑attention using Pennylane variational circuits."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QuantumSelfAttentionGen192:
    """
    Variational quantum circuit that emulates a single‑head self‑attention block.
    The circuit is parameterised by rotation_params (single‑qubit rotations) and
    entangle_params (two‑qubit entangling gates).  The output is a probability
    distribution over the qubits that can be interpreted as attention weights.
    """

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Return a Pennylane QNode that implements the attention circuit."""

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs):
            # Encode inputs as amplitudes
            qml.AmplitudeEmbedding(
                features=inputs,
                wires=range(self.n_qubits),
                normalize=True,
            )

            # Parameterised rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Measure expectation values of Z to obtain attention scores
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the variational circuit and return a probability‑like attention vector.
        Parameters
        ----------
        backend : Any
            Unused – kept for API compatibility with the original interface.
        rotation_params : np.ndarray
            Parameters for single‑qubit rotations. Shape (3 * n_qubits,).
        entangle_params : np.ndarray
            Parameters for two‑qubit entangling gates. Shape (n_qubits - 1,).
        inputs : np.ndarray
            Input vector of length n_qubits to encode into amplitudes.
        shots : int, optional
            Number of shots; ignored for the default simulator but kept for API compatibility.
        Returns
        -------
        np.ndarray
            Normalised attention weights of shape (n_qubits,).
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        raw = circuit(inputs)

        # Convert expectation values to positive scores
        scores = np.array(raw)  # shape (n_qubits,)
        scores = np.clip(scores, a_min=0, a_max=None)

        # Normalise to sum to one
        if scores.sum() == 0:
            scores = np.ones_like(scores) / len(scores)
        else:
            scores = scores / scores.sum()

        return scores


def SelfAttention():
    """
    Factory that returns a QuantumSelfAttentionGen192 instance with a default
    8‑qubit device.  The returned object exposes the same ``run`` interface
    as the classical version.
    """
    return QuantumSelfAttentionGen192(n_qubits=8)
