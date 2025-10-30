import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class SelfAttention:
    """
    Quantum self‑attention module implemented with a variational circuit.
    The circuit generates a learnable attention mask that is applied to
    the input embeddings.  The interface mirrors the classical version
    so that the two implementations can be swapped seamlessly.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits used to encode the attention mask.
    device : qml.Device, optional
        Pennylane device; if ``None`` a default ``default.qubit`` device
        with ``shots=1024`` is created.
    """

    def __init__(self, n_qubits: int = 4, device: qml.Device | None = None):
        self.n_qubits = n_qubits
        if device is None:
            self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024)
        else:
            self.dev = device

        # Parameter shape: one rotation per qubit
        self.param_shape = (n_qubits, 3)

    def _variational_layer(self, params: np.ndarray):
        """Apply a parameter‑shaped rotation layer followed by a CNOT ladder."""
        for i, (theta_x, theta_y, theta_z) in enumerate(params):
            qml.RX(theta_x, wires=i)
            qml.RY(theta_y, wires=i)
            qml.RZ(theta_z, wires=i)

        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    def _circuit(self, params: np.ndarray, embed_slice: np.ndarray):
        """Variational circuit that encodes the inputs and produces a mask."""
        # Encode each input embedding as a rotation on the corresponding qubit
        for i, val in enumerate(embed_slice):
            qml.RY(val, wires=i)

        self._variational_layer(params)

        # Return expectation values of Pauli‑Z as mask components
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Generate a quantum‑derived attention mask and apply it to ``inputs``.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the variational rotation layer.
        entangle_params : np.ndarray
            Parameters for the entangling CNOT ladder (ignored in this
            implementation but kept for API compatibility).
        inputs : np.ndarray
            Input embeddings of shape (batch, seq_len, embed_dim).  Only the
            first ``n_qubits`` values of each sequence are used to encode the
            qubits; the remaining values are ignored.
        shots : int, optional
            Number of shots used by the device.

        Returns
        -------
        np.ndarray
            Masked embeddings of shape (batch, seq_len, embed_dim) where each
            embedding is multiplied element‑wise by the quantum mask.
        """
        batch, seq_len, embed_dim = inputs.shape
        # Ensure device has the requested number of shots
        if hasattr(self.dev, "shots"):
            self.dev.shots = shots

        # Prepare the circuit
        @qml.qnode(self.dev)
        def circuit(params, embed_slice):
            return self._circuit(params, embed_slice)

        # Compute mask for each sequence in the batch
        masks = []
        for b in range(batch):
            # Use the first ``n_qubits`` values of the first token as a simple
            # encoding of the sequence; more elaborate encodings can be added.
            embed_slice = inputs[b, 0, :self.n_qubits]
            mask_vals = circuit(rotation_params, embed_slice)
            # Convert expectation values from [-1, 1] to [0, 1] and reshape
            mask = (np.array(mask_vals) + 1.0) / 2.0
            masks.append(mask)

        masks = np.stack(masks, axis=0)  # (batch, n_qubits)

        # Broadcast mask to all tokens and embed_dim
        mask_expanded = masks[:, :, None]  # (batch, n_qubits, 1)
        mask_expanded = np.repeat(mask_expanded, embed_dim, axis=2)  # (batch, n_qubits, embed_dim)

        # Apply mask to the first ``n_qubits`` positions of each token
        outputs = inputs.copy()
        outputs[:, :, :self.n_qubits] = outputs[:, :, :self.n_qubits] * mask_expanded

        return outputs

__all__ = ["SelfAttention"]
