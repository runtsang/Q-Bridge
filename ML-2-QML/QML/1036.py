"""Quantum self‑attention implementation using Pennylane.

The quantum block mirrors the classical interface but uses a variational circuit
to generate attention weights.  It supports a configurable number of qubits
corresponding to the number of attention heads.  The circuit consists of
parameterised rotations followed by controlled‑X entangling gates whose
strength is set by ``entangle_params``.  The output is the probability
distribution over the measurement basis, which can be interpreted as attention
weights for each head.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Optional

__all__ = ["SelfAttention"]


class SelfAttention:
    """
    Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (heads).  Must match the dimensionality of the
        attention output.
    device : str, default "default.qubit"
        Pennylane device name.
    """

    def __init__(self, n_qubits: int, device: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.device, interface="autograd")
        def circuit(rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
            # Parameterised single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entangling CRX gates
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            # Measure expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for each qubit (length 3 * n_qubits).
        entangle_params : np.ndarray
            Entanglement angles (length n_qubits - 1).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).  For the
            quantum block this is ignored but kept for API compatibility.
        shots : int, default 1024
            Number of shots for the simulation.

        Returns
        -------
        np.ndarray
            Attention weights as a probability distribution over the qubits.
        """
        # The circuit returns expectation values in [-1, 1].
        # Convert them to a probability distribution.
        raw = self.circuit(rotation_params, entangle_params)
        probs = (np.array(raw) + 1.0) / 2.0  # map to [0, 1]
        probs = probs / probs.sum()  # normalize
        return probs
