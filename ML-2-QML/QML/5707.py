import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class HybridEstimator:
    """Quantum variational regressor with a two‑qubit circuit.

    The circuit consists of a single qubit for the input feature and a
    second qubit that acts as a trainable parameter.  The input is encoded
    via an Ry rotation and the trainable parameter via an Rz rotation.
    An entangling CNOT gate allows the model to learn correlations.
    The observable is the Pauli‑Y operator on the second qubit.

    The class exposes a ``predict`` method that returns the expectation
    value and can be used as a drop‑in replacement for a classical
    estimator in training loops.
    """

    def __init__(self,
                 num_qubits: int = 2,
                 trainable_params: int | None = None,
                 observables: list | None = None,
                 device_name: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.device = qml.device(device_name, wires=num_qubits)

        self.input_params = [qml.Symbol("x")]
        self.weight_params = [qml.Symbol(f"w{i}") for i in range(trainable_params or 1)]

        self.observables = observables or [qml.PauliY(1)]

        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.device, interface="autograd")
        def circuit(x, *w):
            qml.RY(x, wires=0)
            for wi in w:
                qml.RZ(wi, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(obs) for obs in self.observables]

        self.circuit = circuit

    def predict(self, x: float | np.ndarray) -> np.ndarray:
        if isinstance(x, (int, float)):
            x = np.array([x])
        return np.array([self.circuit(xi, *self.weight_params) for xi in x])

    def set_weights(self, weights: np.ndarray) -> None:
        self.weight_params = list(weights)

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        return self.predict(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weights={len(self.weight_params)})"

__all__ = ["HybridEstimator"]
