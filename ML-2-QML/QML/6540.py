import pennylane as qml
import numpy as np

class SamplerQNN:
    """
    Parameterized quantum sampler network using Pennylane.
    """
    def __init__(self, num_qubits: int = 2, layers: int = 2):
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Input encoding
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            # Parameterized layers
            for l in range(self.layers):
                for i in range(self.num_qubits):
                    qml.RY(weights[l, i], wires=i)
                # Entangling block
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                qml.CNOT(wires=[self.num_qubits-1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        return circuit

    def __call__(self, inputs: np.ndarray, weights: np.ndarray):
        return self.circuit(inputs, weights)
