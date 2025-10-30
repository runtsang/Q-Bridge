import pennylane as qml
import numpy as np

class SelfAttentionModule:
    """
    Variational self‑attention circuit built with Pennylane.
    The circuit is parameterised by rotation and entanglement angles.
    The ``run`` method mimics the seed API, returning expectation values
    of Pauli‑Z on each qubit.
    """
    def __init__(self, num_qubits: int, num_layers: int = 2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = qml.device("default.qubit", wires=num_qubits)

    def _circuit(self, rotation_params, entangle_params):
        for layer in range(self.num_layers):
            for q in range(self.num_qubits):
                qml.RX(rotation_params[layer, q, 0], wires=q)
                qml.RY(rotation_params[layer, q, 1], wires=q)
                qml.RZ(rotation_params[layer, q, 2], wires=q)
            for q in range(self.num_qubits - 1):
                qml.CZ(wires=[q, q + 1])
                qml.RZ(entangle_params[layer, q], wires=[q, q + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]

    def forward(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """
        Standard forward pass returning expectations.
        """
        qnode = qml.QNode(self._circuit, self.device)
        return np.array(qnode(rotation_params, entangle_params))

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """
        Compatibility wrapper matching the original seed API.
        ``backend`` is ignored – Pennylane’s device is used.
        Returns a dictionary of expectation values for each qubit.
        """
        rotation_params = rotation_params.reshape(self.num_layers, self.num_qubits, 3)
        entangle_params = entangle_params.reshape(self.num_layers, self.num_qubits - 1)
        qnode = qml.QNode(self._circuit, self.device)
        expectations = qnode(rotation_params, entangle_params)
        return {f"qubit_{i}_exp_z": exp for i, exp in enumerate(expectations)}

__all__ = ["SelfAttentionModule"]
