import pennylane as qml
import numpy as np

class EnhancedSamplerQNN:
    """Quantum sampler network using a variational circuit with parameter‑shift support."""
    def __init__(self, num_qubits: int = 2, device: str = "default.qubit", shots: int = 1000):
        self.num_qubits = num_qubits
        self.device = qml.device(device, wires=num_qubits, shots=shots)
        self.input_params = np.random.uniform(0, 2*np.pi, num_qubits)
        self.weight_params = np.random.uniform(0, 2*np.pi, num_qubits * 2)

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs, weights):
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[0, 1])
            for i in range(num_qubits):
                qml.RY(weights[i], wires=i)
            qml.CNOT(wires=[0, 1])
            for i in range(num_qubits):
                qml.RY(weights[num_qubits + i], wires=i)
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.circuit(inputs, self.weight_params)

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        probs = self.forward(inputs)
        return np.random.choice(len(probs), size=num_samples, p=probs)

__all__ = ["EnhancedSamplerQNN"]
