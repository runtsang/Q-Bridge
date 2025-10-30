"""Quantum self‑attention using a Pennylane variational circuit."""
import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Variational self‑attention circuit implemented with Pennylane.
    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits used for amplitude encoding.
    num_layers : int, default 2
        Number of parameterised rotation layers.
    """

    def __init__(self, n_qubits: int = 4, num_layers: int = 2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray):
        """
        Build a variational circuit that encodes the input via amplitude embedding,
        then applies rotation and entanglement layers.
        """
        # Encode the classical data
        qml.AmplitudeEmbedding(
            features=inputs,
            wires=range(self.n_qubits),
            normalize=True,
        )
        # Parameterised rotations
        idx = 0
        for _ in range(self.num_layers):
            for i in range(self.n_qubits):
                qml.RX(rotation_params[idx], wires=i)
                qml.RY(rotation_params[idx + 1], wires=i)
                qml.RZ(rotation_params[idx + 2], wires=i)
                idx += 3
            # Entangling CNOT layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Additional entanglement via RZ
            for i, theta in enumerate(entangle_params):
                qml.RZ(theta, wires=i % self.n_qubits)
        # Return expectation values of PauliZ for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray):
        """
        Execute the circuit and return expectation values.
        Parameters
        ----------
        rotation_params : np.ndarray
            Array containing rotation angles for all layers.
        entangle_params : np.ndarray
            Array of additional entanglement angles.
        inputs : np.ndarray
            Classical input vector to be amplitude‑encoded.
        Returns
        -------
        np.ndarray
            Expectation values of shape (n_qubits,).
        """
        qnode = qml.QNode(self._circuit, self.dev, interface="autograd")
        return qnode(rotation_params, entangle_params, inputs)
