import pennylane as qml
import pennylane.numpy as np

class FCL:
    """
    Quantum fully connected layer using a PennyLane variational circuit.
    """
    def __init__(self, n_qubits: int = 1, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=range(n_qubits))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Evaluate the circuit on a batch of parameter sets.
        Thetas should be a 1‑D iterable or array of shape (batch, n_qubits).
        """
        thetas = np.array(thetas, dtype=np.float64)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        return np.array([self.circuit(params) for params in thetas])

    def train_step(self, thetas, targets, lr: float = 0.01, steps: int = 100):
        """
        Perform a simple gradient‑descent training step on the first batch element.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        params = thetas[0].copy()
        for _ in range(steps):
            params, loss = opt.step_and_cost(
                lambda p: np.mean((self.circuit(p) - targets[0]) ** 2), params)
        return loss

__all__ = ["FCL"]
