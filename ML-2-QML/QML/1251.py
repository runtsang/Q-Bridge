"""EstimatorQNNGen: quantum variational regressor using PennyLane."""

import pennylane as qml
import numpy as np

class EstimatorQNNGen:
    """
    A variational quantum circuit that learns a regression mapping from
    real‑valued inputs to a single expectation value.  The network
    supports multiple qubits, configurable depth, and gradient‑based
    training via PennyLane's autograd interface.
    """
    def __init__(
        self,
        n_qubits: int = 2,
        hidden_layers: int = 2,
        obs: str = "Z",
        dev: qml.Device | None = None,
    ):
        self.n_qubits = n_qubits
        self.hidden_layers = hidden_layers
        self.obs = obs
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        self.params = np.random.uniform(0, 2 * np.pi, size=self._n_params())
        self._build_circuit()

    def _n_params(self) -> int:
        # Each variational layer contributes 3 parameters per qubit (RX, RY, RZ)
        return self.hidden_layers * self.n_qubits * 3

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: np.ndarray):
            # Input encoding via RY rotations
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)
            # Variational layers
            idx = 0
            for _ in range(self.hidden_layers):
                for q in range(self.n_qubits):
                    qml.RX(params[idx], wires=q); idx += 1
                    qml.RY(params[idx], wires=q); idx += 1
                    qml.RZ(params[idx], wires=q); idx += 1
                # Entangling pattern
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # Measurement
            return qml.expval(qml.PauliZ(self.n_qubits - 1))
        self.circuit = circuit

    def predict(self, inputs: np.ndarray) -> float:
        return float(self.circuit(inputs, self.params))

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        preds = np.array([self.circuit(x, params) for x in X])
        return np.mean((preds - y) ** 2)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """
        Gradient‑based training using PennyLane's Adam optimizer.
        """
        opt = qml.AdamOptimizer(lr)
        for epoch in range(epochs):
            grads = opt.gradient(lambda p: self._loss(p, X, y), self.params)
            self.params = opt.step(grads, self.params)
            if epoch % 10 == 0:
                loss = self._loss(self.params, X, y)
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

    @classmethod
    def from_config(cls, config: dict):
        """
        Instantiate from a configuration dictionary.
        """
        return cls(
            n_qubits=config.get("n_qubits", 2),
            hidden_layers=config.get("hidden_layers", 2),
            obs=config.get("obs", "Z"),
            dev=config.get("dev", None),
        )

__all__ = ["EstimatorQNNGen"]
