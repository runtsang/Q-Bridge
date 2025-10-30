"""Quantum‑enhanced self‑attention implemented with Pennylane.

The class mirrors the classical API but replaces the attention score
computation with a variational quantum circuit that produces a scalar
weight per token.  The resulting weighted sum of token embeddings is
returned as a vector of the same dimensionality as the input embeddings.
"""

import pennylane as qml
import pennylane.numpy as qnp
import numpy as np

__all__ = ["QuantumSelfAttention"]

class QuantumSelfAttention:
    """
    Quantum self‑attention that learns attention weights via a
    variational circuit.  The circuit uses each token's embedding as
    rotation angles for RX gates, optionally perturbed by a global
    rotation_params vector.  Entanglement is introduced through a
    chain of CNOT gates.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        dev_name: str = "default.qubit",
    ) -> None:
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits (must match the dimensionality of the
            token embeddings).
        n_layers : int
            Number of variational layers applied after the initial
            rotation gates.
        dev_name : str
            Pennylane device name (e.g. "default.qubit" or an Aer
            backend).
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev_name, wires=n_qubits)

        # Build a QNode that returns a vector of expectation values of PauliZ
        @qml.qnode(self.dev, interface="numpy")
        def circuit(rot_params, ent_params):
            # Apply token‑dependent rotations
            for i in range(self.n_qubits):
                qml.RX(rot_params[i], wires=i)
            # Optional global rotation perturbation
            for i in range(self.n_qubits):
                qml.RX(ent_params[i], wires=i)
            # Variational layers
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(0.1, wires=i)  # small rotation as placeholder
                for i in range(self.n_qubits - 1):
                    qml.CNOT(i, i + 1)
            # Measure expectation of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a quantum‑enhanced weighted sum of token embeddings.

        Parameters
        ----------
        rotation_params : np.ndarray
            Global rotation vector of shape (n_qubits,) that is added to
            each token embedding before the circuit is evaluated.
        entangle_params : np.ndarray
            Global entanglement vector of shape (n_qubits,) used as
            additional rotation angles in the circuit.
        inputs : np.ndarray
            Token embeddings of shape (batch, seq, embed_dim) or
            (seq, embed_dim).  The last dimension must equal ``n_qubits``.
        Returns
        -------
        np.ndarray
            Weighted sum of the input embeddings.  For a batched input
            the output shape is (batch, embed_dim); for a single
            sequence it is (embed_dim,).
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        rotation_params = np.asarray(rotation_params, dtype=np.float64)
        entangle_params = np.asarray(entangle_params, dtype=np.float64)

        if rotation_params.shape[0]!= self.n_qubits:
            raise ValueError(f"rotation_params must have length {self.n_qubits}")
        if entangle_params.shape[0]!= self.n_qubits:
            raise ValueError(f"entangle_params must have length {self.n_qubits}")

        # Helper to compute a scalar weight for a single token
        def token_weight(token):
            # token is a vector of length n_qubits
            rot = token + rotation_params
            attn_vec = self.circuit(rot, entangle_params)
            # Convert expectation values from [-1,1] to [0,1] and average
            weight = (np.array(attn_vec) + 1.0).mean() / 2.0
            return weight

        if inputs.ndim == 2:
            # (seq, embed_dim)
            seq, _ = inputs.shape
            weights = np.array([token_weight(tok) for tok in inputs])
            out = np.sum(weights[:, np.newaxis] * inputs, axis=0)
            return out
        elif inputs.ndim == 3:
            # (batch, seq, embed_dim)
            batch, seq, _ = inputs.shape
            outs = []
            for b in range(batch):
                seq_inputs = inputs[b]
                weights = np.array([token_weight(tok) for tok in seq_inputs])
                out = np.sum(weights[:, np.newaxis] * seq_inputs, axis=0)
                outs.append(out)
            return np.stack(outs, axis=0)
        else:
            raise ValueError("inputs must be 2‑D or 3‑D array")
