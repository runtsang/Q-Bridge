import pennylane as qml
import pennylane.numpy as np

class SamplerQNNImpl:
    """
    A variational quantum sampler implemented with Pennylane.
    Uses a 4‑qubit circuit with entangling layers and parameterized rotations.
    """
    def __init__(self, wires=4, device_name="default.qubit"):
        self.wires = wires
        self.dev = qml.device(device_name, wires=self.wires)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: np.ndarray, weights: np.ndarray):
            # Encode classical inputs as RY rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # First entangling layer
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])

            # Parameterized rotation layers
            for i, w in enumerate(weights):
                qml.RY(w, wires=i % self.wires)

            # Second entangling layer
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])

            # Sample two qubits to obtain a 2‑bit distribution
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        return circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray):
        """
        Execute the quantum circuit and return sampled results.
        """
        return self.circuit(inputs, weights)

def SamplerQNN():
    """
    Factory function that returns an instance of the quantum sampler.
    """
    return SamplerQNNImpl()

__all__ = ["SamplerQNN"]
