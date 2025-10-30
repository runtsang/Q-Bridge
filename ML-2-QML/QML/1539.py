"""Quantum fully‑connected layer using Pennylane."""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QuantumFullyConnectedLayer:
    """
    A single‑qubit variational circuit that emulates a fully‑connected layer.
    Parameters are passed to the ``run`` method as a flat vector.
    """

    def __init__(self, n_qubits: int = 1, device: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self.theta = pnp.array(0.0)

    # ------------------------------------------------------------------
    # Variational circuit
    # ------------------------------------------------------------------
    def _circuit(self, theta: float):
        qml.Hadamard(wires=range(self.n_qubits))
        qml.RY(theta, wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, thetas: np.ndarray | Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for each parameter in ``thetas``.
        The method returns a 1‑D numpy array of expectation values.
        """
        thetas = np.asarray(thetas, dtype=np.float32).flatten()
        expectations = np.array([self._circuit(theta) for theta in thetas])
        return expectations

    # ------------------------------------------------------------------
    # Simple training routine (optional)
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.1,
    ) -> None:
        """
        Gradient‑descent training that minimizes mean‑squared error
        between the circuit expectation values and target labels.
        """
        opt = qml.optimize.AdamOptimizer(lr)
        params = pnp.array([0.0], requires_grad=True)

        @qml.qnode(self.device)
        def qnode(theta):
            return self._circuit(theta)

        def loss_fn(p):
            preds = pnp.array([qnode(p[0]) for _ in X])
            return pnp.mean((preds - Y) ** 2)

        for _ in range(epochs):
            params = opt.step(loss_fn, params)

        self.theta = params[0]


def FCL() -> QuantumFullyConnectedLayer:
    """Return a one‑qubit variational fully‑connected layer."""
    return QuantumFullyConnectedLayer()


__all__ = ["QuantumFullyConnectedLayer", "FCL"]
