import pennylane as qml
import numpy as np

class AdvancedSamplerQNN:
    """
    Quantum sampler network using a parameterized ansatz on 2 qubits.
    - Variational circuit with Ry rotations and controlled‑Z entanglement layers.
    - Returns probability distribution over computational basis.
    - Provides sample() method using the simulator backend.
    """
    def __init__(self, dev_name: str = "default.qubit", shots: int = 1024):
        self.dev = qml.device(dev_name, wires=2, shots=shots)
        self.params = np.random.uniform(0, 2*np.pi, size=4)  # weights
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray):
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            qml.CZ(wires=[0, 1])
            qml.RY(self.params[0], wires=0)
            qml.RY(self.params[1], wires=1)
            qml.CZ(wires=[0, 1])
            qml.RY(self.params[2], wires=0)
            qml.RY(self.params[3], wires=1)
            return qml.probs(wires=[0, 1])
        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.circuit(inputs)

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        probs = self.forward(inputs)
        return np.random.choice(a=4, size=num_samples, p=probs)

def SamplerQNN():
    return AdvancedSamplerQNN()

__all__ = ["SamplerQNN", "AdvancedSamplerQNN"]
