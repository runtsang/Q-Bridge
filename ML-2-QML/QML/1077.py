import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Quantum self‑attention implemented with Pennylane.  The circuit
    emulates the classical multi‑head attention by mapping
    `rotation_params` to single‑qubit rotations and
    `entangle_params` to controlled‑RX gates that entangle adjacent
    qubits.  The output is the vector of Z‑expectation values,
    which can be interpreted as attention scores.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits (must match the dimensionality of the
        classical counterpart).
    device : str, optional
        Pennylane device name (e.g. ``default.qubit`` or a
        quantum backend such as ``ibmq_qasm_simulator``).
    shots : int, optional
        Number of measurement shots for the expectation.
    """
    def __init__(self, n_qubits: int = 4, device: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=shots)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(rot_params, ent_params):
            # single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rot_params[3 * i], wires=i)
                qml.RY(rot_params[3 * i + 1], wires=i)
                qml.RZ(rot_params[3 * i + 2], wires=i)

            # entanglement
            for i in range(self.n_qubits - 1):
                qml.CRX(ent_params[i], wires=[i, i + 1])

            # measurement expectation of Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, backend=None, rotation_params: np.ndarray = None,
            entangle_params: np.ndarray = None, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return the expectation values.

        Parameters
        ----------
        backend : str, optional
            Pennylane device name to override the default.
        rotation_params : np.ndarray
            Array of shape (3 * n_qubits,) containing RX/RY/RZ angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) containing CRX angles.
        shots : int, optional
            Number of measurement shots (ignored if backend is a
            differentiable simulator).

        Returns
        -------
        np.ndarray
            Expectation values of shape (n_qubits,).
        """
        if backend is not None:
            self.dev = qml.device(backend, wires=self.n_qubits, shots=shots)
            self._build_qnode()
        return np.array(self.circuit(rotation_params, entangle_params))

__all__ = ["SelfAttention"]
