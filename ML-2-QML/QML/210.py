import pennylane as qml
import torch

class SamplerQNNGen:
    """
    Variational quantum sampler with 3 qubits and parameterized rotation layers.
    """
    def __init__(self, dev=None):
        self.dev = dev or qml.device("default.qubit", wires=3)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # Encode inputs as rotations
            for i in range(3):
                qml.RY(inputs[i], wires=i)
            # Entanglement layer
            qml.CNOT(0, 1)
            qml.CNOT(1, 2)
            # Parameterized rotation layer
            for i in range(3):
                qml.RY(weights[i], wires=i)
            # Second entanglement
            qml.CNOT(0, 1)
            qml.CNOT(1, 2)
            return qml.probs(wires=[0, 1, 2])

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the variational circuit and return probability distribution over 8 basis states.
        """
        return self.circuit(inputs, weights)
