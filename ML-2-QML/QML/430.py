import pennylane as qml
import numpy as np
from typing import Iterable

class FullyConnectedLayer:
    """
    Variational quantum circuit emulating a fully connected layer.
    Uses Pennylane's default.qubit device and the parameter‑shift rule
    for gradient estimation. Provides a simple training routine.
    """
    def __init__(self,
                 n_qubits: int = 1,
                 dev_name: str = "default.qubit",
                 wires: List[int] = None) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)
        self.wires = wires if wires is not None else list(range(n_qubits))
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(params):
            for i, w in enumerate(self.wires):
                qml.Hadamard(w)
                qml.RY(params[i], w)
            return qml.expval(qml.PauliZ(self.wires[0]))
        self.circuit = circuit

    def run(self, thetas: Iterable[Iterable[float]]) -> np.ndarray:
        """
        Evaluate expectation values for a batch of parameter vectors.
        """
        return np.array([self.circuit(np.array(t, dtype=np.float32))
                         for t in thetas])

    def train(self,
              data: np.ndarray,
              target: np.ndarray,
              lr: float = 0.01,
              epochs: int = 100) -> np.ndarray:
        """
        Simple gradient descent training using the parameter‑shift rule.
        """
        opt = qml.GradientDescentOptimizer(lr)
        params = np.random.uniform(-np.pi, np.pi, size=self.n_qubits)
        for _ in range(epochs):
            cost = np.mean((self.circuit(params) - target) ** 2)
            params = opt.step(lambda p: np.mean((self.circuit(p) - target) ** 2),
                              params)
        return params

__all__ = ["FullyConnectedLayer"]
