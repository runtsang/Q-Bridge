"""Quantum self‑attention module using Pennylane variational circuits.

The module implements a multi‑qubit circuit that encodes the input
through parameterised rotations and entangling gates.  The circuit
produces a probability distribution that is interpreted as attention
weights.  The quantum block can be fused with the classical module
to form a hybrid attention layer.

The interface matches the classical version: ``run`` accepts
``rotation_params``, ``entangle_params`` and ``inputs`` and returns
a NumPy array of the same shape as the input.
"""

import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as pnp

class QuantumSelfAttentionModule:
    """Variational quantum self‑attention."""

    def __init__(self, n_qubits: int = 4, shots: int = 1024, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device(device_name, wires=n_qubits, shots=shots)

        # Parameter shapes
        self.rotation_shape = (n_qubits, 3)   # RX, RY, RZ per qubit
        self.entangle_shape = (n_qubits - 1,)  # CX between consecutive qubits

        self._build_circuit()

    def _build_circuit(self):
        """Build a Pennylane QNode that returns a probability vector over all basis states."""

        @qml.qnode(self.dev, interface="torch")
        def circuit(rot_params, ent_params, inputs):
            # Encode the classical input into the first qubit via a rotation
            for i in range(self.n_qubits):
                qml.RX(rot_params[i, 0], wires=i)
                qml.RY(rot_params[i, 1], wires=i)
                qml.RZ(rot_params[i, 2], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(ent_params[i], wires=[i, i + 1])

            # Measurement: return probability distribution
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3) – RX, RY, RZ angles per qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) – CX rotation angles.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).  Only the first
            element of each sequence is used to seed the circuit; the rest are
            ignored to keep the interface compatible.

        Returns
        -------
        output : np.ndarray
            A probability vector of shape (batch, seq_len, n_qubits) that
            serves as attention scores.  The vector is interpreted as
            soft‑maxed weights over the qubits.
        """
        batch, seq_len, _ = inputs.shape
        outputs = []

        for b in range(batch):
            batch_out = []
            for s in range(seq_len):
                # Use the first token embedding to influence the rotation
                # params – simple linear mapping for demonstration.
                rot = rotation_params + 0.1 * inputs[b, s, :self.n_qubits]
                ent = entangle_params

                probs = self.circuit(rot, ent, inputs[b, s, :])
                # Convert to attention weights
                attn = probs / probs.sum()
                batch_out.append(attn.detach().numpy())

            outputs.append(np.stack(batch_out, axis=0))

        return np.stack(outputs, axis=0)

def SelfAttention():
    """Factory that returns a ready‑to‑use quantum attention module."""
    return QuantumSelfAttentionModule(n_qubits=4, shots=1024, device_name="default.qubit")

__all__ = ["QuantumSelfAttentionModule", "SelfAttention"]
