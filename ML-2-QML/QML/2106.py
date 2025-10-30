import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class FullyConnectedLayer:
    """Parameterized quantum circuit mimicking a fully connected layer."""
    def __init__(self, n_qubits: int = 1, device_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit with given parameters."""
        exp = self.circuit(thetas)
        return np.array([exp])

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """Compute gradient of the expectation w.r.t. parameters."""
        grad_fn = qml.grad(self.circuit)
        grad = grad_fn(thetas)
        return np.array(grad)

    def get_params(self) -> np.ndarray:
        """Return current circuit parameters (placeholder)."""
        return np.array([])

__all__ = ["FullyConnectedLayer"]
