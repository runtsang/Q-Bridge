import pennylane as qml
import pennylane.numpy as np

class SamplerQNN:
    """
    A quantum sampler based on a variational circuit with 3 qubits.
    The circuit consists of an input embedding, a single rotation layer,
    and a CNOT entanglement pattern.  The :meth:`forward` method returns
    the probability of measuring each computational basis state.
    """

    def __init__(self, n_qubits: int = 3, seed: int = 42):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024, seed=seed)
        self.weights = np.random.randn(n_qubits) * 0.1

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for i in range(n_qubits):
                qml.RZ(weights[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate the circuit and return the probability distribution.
        """
        if weights is None:
            weights = self.weights
        return self.circuit(inputs, weights)

    def sample(self, inputs: np.ndarray, weights: np.ndarray | None = None) -> int:
        """
        Sample a single basis state from the probability distribution.
        """
        probs = self.forward(inputs, weights)
        return np.random.choice(len(probs), p=probs)

__all__ = ["SamplerQNN"]
