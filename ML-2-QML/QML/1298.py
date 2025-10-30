import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class HybridSelfAttentionQML:
    """Variational self‑attention circuit built with PennyLane.

    The circuit uses a series of parameterized rotations, a fixed entangling
    pattern (CNOT chain), and additional RX gates to emulate key/value
    projections.  The output is a vector of expectation values that can be
    interpreted as attention logits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match the dimensionality of the input).
    num_layers : int, default 2
        Number of rotation/entanglement layers in the circuit.
    """

    def __init__(self, n_qubits: int, num_layers: int = 2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Build a single forward pass of the variational circuit."""
        # Layered rotations
        for layer in range(self.num_layers):
            for i in range(self.n_qubits):
                idx = 3 * layer * self.n_qubits + 3 * i
                qml.Rot(
                    rotation_params[idx],
                    rotation_params[idx + 1],
                    rotation_params[idx + 2],
                    wires=i,
                )
            # Entangling CNOT chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Additional RX gates to emulate entangle_params
            for i in range(self.n_qubits):
                qml.RX(entangle_params[layer * self.n_qubits + i], wires=i)

        # Return expectation values of PauliZ as logits
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit on the chosen backend.

        Parameters
        ----------
        rotation_params : ndarray
            Array of shape (3 * num_layers * n_qubits,).
        entangle_params : ndarray
            Array of shape (num_layers * n_qubits,).
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        ndarray
            Expectation values of PauliZ for each qubit.
        """
        self.dev.shots = shots
        return self.qnode(rotation_params, entangle_params)


__all__ = ["HybridSelfAttentionQML"]
