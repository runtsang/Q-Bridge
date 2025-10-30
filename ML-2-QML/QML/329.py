import pennylane as qml
import numpy as np

class FCL:
    """
    Variational quantum circuit that mimics a fully‑connected layer.
    Supports up to two qubits, entanglement, and gradient‑based training.
    The API mirrors the classical version: `run` returns a 1‑D array of
    expectation values for each supplied parameter set.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self.params = np.random.randn(n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(theta):
            # Simple Ry ansatz with optional entanglement
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(theta[i], wires=i)
            if self.n_qubits > 1:
                qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for each theta in the input array.
        Returns a 1‑D numpy array of expectation values.
        """
        expect = []
        for theta in thetas:
            # Broadcast the single parameter to all qubits
            theta_vec = np.full(self.n_qubits, theta)
            expect.append(self.circuit(theta_vec))
        return np.array(expect)

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200):
        """
        Gradient‑descent training using Pennylane's autograd interface.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)

        for _ in range(epochs):
            for x, target in zip(X, y):
                def cost(theta):
                    pred = self.circuit(theta)
                    return (pred - target)**2
                self.params = opt.step(cost, self.params)
