import pennylane as qml
import numpy as np

class SamplerQNNExtended:
    """
    A quantum sampler network that extends the original 2‑qubit circuit.
    Uses a 3‑qubit variational circuit with entangling layers and parameter‑shift gradient support.
    """
    def __init__(self, n_qubits: int = 3, n_layers: int = 2, seed: int | None = None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(0, 2*np.pi, (n_layers, n_qubits))
        self.input_params = rng.uniform(0, 2*np.pi, n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: np.ndarray, weights: np.ndarray):
            # Encode inputs as Ry rotations
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[l, i], wires=i)
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])  # wrap‑around
            # Measurement: return probabilities of computational basis
            return qml.probs(wires=range(self.n_qubits))
        self.circuit = circuit

    def sample(self, inputs: np.ndarray, n_shots: int = 1000) -> np.ndarray:
        """
        Generate samples from the quantum circuit.
        """
        probs = self.circuit(inputs, self.params)
        probs = probs.detach().numpy()
        outcomes = np.random.choice(len(probs), size=n_shots, p=probs)
        bitstrings = [format(out, f"0{self.n_qubits}b") for out in outcomes]
        return np.array(bitstrings)

    def expectation(self, observable: qml.operation.Operator) -> float:
        """
        Compute expectation value of a given observable.
        """
        @qml.qnode(self.dev, interface="torch")
        def exp_circuit(inputs, weights):
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[l, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(observable)
        return exp_circuit(self.input_params, self.params).item()

__all__ = ["SamplerQNNExtended"]
