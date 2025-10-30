import numpy as np
import pennylane as qml
from pennylane.optimize import GradientDescentOptimizer

class FCL:
    """
    Parameterised quantum circuit that emulates a fully‑connected layer.

    The circuit uses a stack of RY‑RZ alternating layers and measures the
    Z expectation on each qubit.  The number of qubits equals the input
    dimensionality.  The ``run`` method evaluates the circuit on a given
    parameter vector and returns a NumPy array of expectation values.
    A ``train`` helper implements gradient‑based optimisation using
    PennyLane’s autograd capabilities.
    """

    def __init__(self,
                 n_qubits: int,
                 depth: int = 1,
                 dev: str | None = None,
                 shots: int = 1000) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.dev = qml.device(dev or "default.qubit", wires=n_qubits, shots=shots)
        self.params = np.random.randn(depth, n_qubits, 2)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd", diff_method="parameter-shift")
        def circuit(params):
            for d in range(self.depth):
                for i in range(self.n_qubits):
                    qml.RY(params[d, i, 0], wires=i)
                    qml.RZ(params[d, i, 1], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        params = np.array(thetas, dtype=np.float32).reshape(self.depth, self.n_qubits, 2)
        return self.circuit(params)

    def train(self,
              data: Iterable[Iterable[float]],
              targets: Iterable[float],
              epochs: int = 100,
              lr: float = 0.01,
              verbose: bool = False) -> None:
        opt = GradientDescentOptimizer(stepsize=lr)
        data = list(data)
        targets = np.array(targets, dtype=np.float32)
        for epoch in range(epochs):
            self.params = opt.step(self.circuit, self.params)
            preds = np.array([self.run(d).sum() for d in data], dtype=np.float32)
            loss = ((preds - targets) ** 2).mean()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} loss={loss:.4f}")

    def __repr__(self) -> str:
        return f"FCL(n_qubits={self.n_qubits}, depth={self.depth}, shots={self.shots})"

__all__ = ["FCL"]
