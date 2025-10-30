import pennylane as qml
import numpy as np

class SelfAttentionQML:
    """
    Variational self‑attention implemented with PennyLane.
    The circuit outputs an attention matrix via expectation values of
    Pauli‑Z on pairs of qubits, which is then used to weight the input values.
    """

    def __init__(self, n_qubits: int, num_layers: int = 2, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=n_qubits)

        # Parameter shapes
        self.rotation_shape = (n_qubits, 3)
        self.entangle_shape = (n_qubits - 1,)

        # Build the variational circuit
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev)
        def circuit(rotation_params, entangle_params):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.RX(rotation_params[i, 0], wires=i)
                qml.RY(rotation_params[i, 1], wires=i)
                qml.RZ(rotation_params[i, 2], wires=i)

            # Entangling CRX
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Measure expectation values of Z⊗Z for adjacent pairs
            exp_vals = []
            for i in range(self.n_qubits - 1):
                exp_vals.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
            return exp_vals

        self.circuit = circuit

    def run(self, inputs: np.ndarray, rotation_params: np.ndarray,
            entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Compute attention weights from quantum measurements and apply them
        to the input sequence.

        Parameters
        ----------
        inputs : np.ndarray
            Input sequence of shape (seq_len, embed_dim).
        rotation_params : np.ndarray
            Parameters for rotation gates, shape (n_qubits, 3).
        entangle_params : np.ndarray
            Parameters for entangling gates, shape (n_qubits-1,).
        shots : int, optional
            Number of shots for the simulator.

        Returns
        -------
        np.ndarray
            Weighted sum of the inputs according to the quantum‑derived attention.
        """
        # Quantum circuit returns expectation values in [-1,1]; map to [0,1]
        raw = self.circuit(rotation_params, entangle_params, shots=shots)
        attn_weights = (np.array(raw) + 1) / 2  # shape (n_qubits-1,)

        # Pad to match seq_len
        seq_len = inputs.shape[0]
        if seq_len!= self.n_qubits:
            raise ValueError("Input sequence length must match n_qubits")

        # Apply attention weights to values
        values = inputs
        weighted = values * attn_weights[:, None]
        return weighted.sum(axis=0)

__all__ = ["SelfAttentionQML"]
