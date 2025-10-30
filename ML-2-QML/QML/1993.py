"""Quantum implementation of a fully connected layer using a parameterised circuit."""

import pennylane as qml
import numpy as np

class FullyConnectedLayer:
    """
    Parameterised quantum circuit that mimics a single‑unit fully connected layer.
    The circuit consists of a Hadamard layer, a parameterised Ry rotation on each qubit
    and a measurement of the Pauli‑Z expectation of a single output qubit.
    The interface matches the legacy `run` method and includes a lightweight training helper.
    """
    def __init__(self, n_qubits: int = 1, dev_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)
        # initial parameters (unused in the current circuit but kept for extensibility)
        self._params = np.random.uniform(0, 2 * np.pi, size=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas):
            for i in range(self.n_qubits):
                qml.RY(thetas[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Legacy compatible run: returns a 1‑D array with the expectation value."""
        return np.array([self.circuit(thetas)])

    def train_on(self, X: np.ndarray, y: np.ndarray, epochs: int = 200,
                 lr: float = 0.01, verbose: bool = False) -> None:
        """
        Very small training helper that optimises the parameters against a mean‑squared‑error loss.
        Training is performed with the autograd interface of PennyLane.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        params = self._params.copy()

        for epoch in range(epochs):
            def cost(p):
                preds = np.array([self.circuit(X[i]) for i in range(len(X))])
                return np.mean((preds - y) ** 2)

            params, cost_val = opt.step_and_cost(cost, params)
            if verbose and epoch % max(1, (epochs // 5)) == 0:
                print(f"Epoch {epoch:04d} loss={cost_val:.4f}")
        self._params = params

def FCL():
    """Return the FullyConnectedLayer class for backward compatibility."""
    return FullyConnectedLayer

__all__ = ["FullyConnectedLayer", "FCL"]
