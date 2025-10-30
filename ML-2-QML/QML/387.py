"""Quantum self‑attention implemented with PennyLane."""
import numpy as np
import pennylane as qml

class SelfAttentionUnit:
    """Quantum self‑attention block using a parameterized circuit."""
    def __init__(self, n_qubits: int = 4, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray, target_wire: int):
        """
        Build a single‑qubit attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3) – RX, RY, RZ angles per qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) – CRX angles for nearest‑neighbour entanglement.
        inputs : np.ndarray
            Shape (n_qubits,) – amplitude‑encoded data vector.
        target_wire : int
            Wire whose expectation is returned as the attention weight.
        """
        for i in range(self.n_qubits):
            qml.RX(rotation_params[i, 0], wires=i)
            qml.RY(rotation_params[i, 1], wires=i)
            qml.RZ(rotation_params[i, 2], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        qml.StatePrep(inputs / np.linalg.norm(inputs), wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(target_wire))

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the attention circuit for each qubit and return a probability‑based attention vector.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3).
        entangle_params : np.ndarray
            Shape (n_qubits-1,).
        inputs : np.ndarray
            Shape (n_qubits,).
        shots : int, optional
            Number of shots for sampling (ignored on state‑vector device).

        Returns
        -------
        np.ndarray
            Attention weights of shape (n_qubits,).
        """
        if inputs.shape[0]!= self.n_qubits:
            raise ValueError("Input length must match number of qubits.")
        weights = []
        for wire in range(self.n_qubits):
            circuit = qml.QNode(
                lambda rp, ep, inp, w=wire: self._circuit(rp, ep, inp, w),
                self.dev,
                interface="autograd",
            )
            expval = circuit(rotation_params, entangle_params, inputs)
            weights.append(expval)
        weights = np.array(weights)
        # Convert raw expectations to a probability distribution
        probs = np.exp(weights) / np.sum(np.exp(weights))
        return probs

__all__ = ["SelfAttentionUnit"]
