import pennylane as qml
import pennylane.numpy as np

class AdvancedSamplerQNN:
    """
    Variational quantum sampler with configurable depth and entanglement.
    Produces probability amplitudes via a parameterized circuit.
    """
    def __init__(self, num_qubits=2, depth=2, device="default.qubit", shots=1024):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.dev = qml.device(device, wires=num_qubits, shots=shots)

    def circuit(self, inputs, weights):
        """
        Build a depth‑d variational circuit.
        :param inputs: array of shape (num_qubits,) – rotation angles for input encoding.
        :param weights: array of shape (num_qubits*depth*2,) – rotation angles for each layer.
        :return: probability distribution over computational basis states.
        """
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)

        idx = 0
        for d in range(self.depth):
            for i in range(self.num_qubits):
                qml.RY(weights[idx], wires=i)
                idx += 1
            # entangling layer (ring topology)
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

        return qml.probs(wires=range(self.num_qubits))

    def __call__(self, inputs, weights):
        """Execute the circuit and return the probability distribution."""
        probs = self.circuit(inputs, weights)
        return probs

__all__ = ["AdvancedSamplerQNN"]
