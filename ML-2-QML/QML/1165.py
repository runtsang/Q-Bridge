import pennylane as qml
import pennylane.numpy as np

__all__ = ["ConvGen102QML", "Conv"]

class ConvGen102QML:
    """Parameter‑efficient variational quanvolution layer."""
    def __init__(
        self,
        kernel_size: int = 2,
        device: str = "default.qubit",
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.dev = qml.device(device, wires=self.n_qubits, shots=shots)

        # Random initial parameters for a 2‑layer ansatz
        self.params = np.random.uniform(0, 2 * np.pi, (self.n_qubits, 3))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Data encoding: X gate if pixel > threshold
            for i, val in enumerate(inputs):
                if val > self.threshold:
                    qml.PauliX(i)

            # Variational ansatz: 2‑layer RY–RZ–RX with CNOTs
            for layer in range(2):
                for i in range(self.n_qubits):
                    qml.RY(params[i, 0], wires=i)
                    qml.RZ(params[i, 1], wires=i)
                    qml.RX(params[i, 2], wires=i)
                    if i < self.n_qubits - 1:
                        qml.CNOT(i, i + 1)

            # Return expectation values of PauliZ on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data) -> float:
        """Evaluate the variational circuit on a single kernel image."""
        data_arr = np.array(data).reshape(self.n_qubits)
        expvals = self.circuit(data_arr, self.params)
        probs = (1 - expvals) / 2  # Convert <Z> to probability of |1>
        return probs.mean().item()

    def train(self, data, labels, lr: float = 0.01, epochs: int = 100):
        """Simple gradient descent to fit the circuit to labels."""
        opt = qml.GradientDescentOptimizer(lr)

        @qml.qnode(self.dev, interface="autograd")
        def cost_fn(params: np.ndarray) -> float:
            expvals = self.circuit(data, params)
            probs = (1 - np.array(expvals)) / 2
            return np.mean((probs - labels) ** 2)

        for _ in range(epochs):
            self.params = opt.step(cost_fn, self.params)
        return self.params

def Conv() -> ConvGen102QML:
    """Factory that returns a ConvGen102QML instance with default configuration."""
    return ConvGen102QML()
