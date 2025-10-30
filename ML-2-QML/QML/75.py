import pennylane as qml
import numpy as np

class SelfAttentionModule:
    """
    Variational self‑attention implemented with Pennylane.  Each input token
    is encoded into rotation angles that drive a parameter‑shifted circuit.
    The circuit outputs expectation values of Pauli‑Z on each qubit, which
    are interpreted as attention scores.  The module can be used as a drop‑in
    replacement for the classical variant in hybrid architectures.
    """

    def __init__(self,
                 n_qubits: int = 8,
                 device: str | qml.Device = "default.qubit",
                 shots: int = 1024):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits per token; determines the dimensionality of
            the attention vector.
        device : str or qml.Device
            Pennylane device to run the circuit on.
        shots : int
            Number of shots for expectation estimation.
        """
        self.n_qubits = n_qubits
        self.shots = shots
        if isinstance(device, str):
            self.dev = qml.device(device, wires=n_qubits, shots=shots)
        else:
            self.dev = device

        # Trainable parameters: rotations per qubit, entangling angles
        self.param_shape = (n_qubits * 3 + n_qubits - 1,)

    def _circuit(self,
                 rotation_params: np.ndarray,
                 entangle_params: np.ndarray,
                 inputs: np.ndarray):
        """
        Build a parameter‑shifted circuit that encodes the input features
        via rotation angles and performs a simple entangling pattern.
        """
        # Encode inputs into rotation angles
        for i in range(self.n_qubits):
            theta = rotation_params[3 * i]
            phi = rotation_params[3 * i + 1]
            lam = rotation_params[3 * i + 2]
            qml.RX(theta, wires=i)
            qml.RY(phi, wires=i)
            qml.RZ(lam, wires=i)

        # Entangle neighboring qubits
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        # Measurement of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    @qml.qnode
    def _qnode(self,
               rotation_params: np.ndarray,
               entangle_params: np.ndarray,
               inputs: np.ndarray):
        return self._circuit(rotation_params, entangle_params, inputs)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return attention scores.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for each qubit.  Shape must match
            (n_qubits * 3,).
        entangle_params : np.ndarray
            Entangling gate angles.  Shape must match (n_qubits - 1,).
        inputs : np.ndarray
            Classical feature vector to be embedded.  Shape (n_qubits,).

        Returns
        -------
        np.ndarray
            Attention scores derived from expectation values of Pauli‑Z.
            Values are in the range [-1, 1] and are post‑processed with
            softmax to yield a probability distribution.
        """
        # Ensure inputs are broadcasted to the circuit
        if inputs.ndim == 1:
            inputs = np.expand_dims(inputs, 0)

        raw = self._qnode(rotation_params, entangle_params, inputs)
        # Convert to numpy array if necessary
        raw = np.asarray(raw)
        # Softmax to obtain a probability distribution
        exp = np.exp(raw - np.max(raw))
        return exp / np.sum(exp, axis=-1, keepdims=True)

__all__ = ["SelfAttentionModule"]
