"""Variational quantum regressor with parameterised rotations and entanglement.

The circuit operates on a configurable number of qubits, applies
data‑encoding rotations, a stack of trainable layers, and measures
the expectation of Pauli‑Z on each qubit.  The class exposes the
same public API as the classical version for compatibility."""
import pennylane as qml
import pennylane.numpy as np


class EstimatorQNN:
    """
    Variational quantum neural network for regression.

    Parameters
    ----------
    num_qubits : int, default 4
        Number of qubits in the circuit.
    layers : int, default 3
        Depth of the parameterised rotation layers.
    entanglement : str, default 'cnot'
        Pattern of entanglement; currently supports 'cnot' (nearest‑neighbour).
    """

    def __init__(self, num_qubits: int = 4, layers: int = 3,
                 entanglement: str = "cnot") -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Initialise random weights: shape (layers, qubits, 3) for RX, RY, RZ
        self.weights = np.random.randn(layers, num_qubits, 3)

        # Observable: Pauli‑Z on each qubit
        self.obs = [qml.PauliZ(i) for i in range(num_qubits)]

        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> list[float]:
        """Build the variational circuit."""
        # Data‑encoding: RX per qubit
        for i in range(self.num_qubits):
            qml.RX(inputs[i], wires=i)

        # Parameterised rotation layers with entanglement
        for l in range(self.layers):
            for i in range(self.num_qubits):
                qml.Rot(*weights[l, i], wires=i)
            if self.entanglement == "cnot":
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

        return [qml.expval(o) for o in self.obs]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return expectation values."""
        return self.qnode(inputs, self.weights)

    def loss(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """Mean‑squared‑error loss for quantum predictions."""
        return np.mean((preds - targets) ** 2)


__all__ = ["EstimatorQNN"]
