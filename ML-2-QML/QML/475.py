import pennylane as qml
import numpy as np

class HybridEstimator:
    """
    A parameterized variational quantum circuit with entangling layers
    and an expectation-value readout.  The circuit is designed to be
    compatible with Pennylane's autograd interface, enabling joint
    training with classical optimizers.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 n_layers: int = 3,
                 seed: int | None = None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal((n_layers, n_qubits, 3))
        self.input_params = rng.standard_normal(n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray,
                    weights: np.ndarray) -> float:
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.Rot(*weights[layer, qubit], wires=qubit)
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> float:
        return self.circuit(inputs, self.weights)

__all__ = ["HybridEstimator"]
