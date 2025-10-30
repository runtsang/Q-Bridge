import pennylane as qml
import pennylane.numpy as np
import torch

class HybridSamplerQNN:
    """
    Quantum sampler neural network implemented with PennyLane.
    Mirrors the classical API: `forward` returns a probability distribution,
    and `sample` draws classical samples from that distribution.
    """
    def __init__(self, num_qubits: int = 2, num_layers: int = 2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.weight_params = np.random.uniform(0, 2 * np.pi, size=(num_layers, num_qubits))
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, inputs, weights):
        # Encode classical inputs with Ry rotations
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)

        # Variational layers with entanglement
        for layer in range(self.num_layers):
            for q in range(self.num_qubits):
                qml.RY(weights[layer, q], wires=q)
            for q in range(self.num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

        # Return full probability distribution
        return qml.probs(wires=range(self.num_qubits))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the circuit and return a 2‑class probability distribution
        by marginalizing over the second qubit.
        """
        probs = self.qnode(inputs.numpy(), self.weight_params)
        # Collapse to 2‑dim distribution: |00>+|10> vs |01>+|11>
        probs_2d = np.array([probs[0] + probs[2], probs[1] + probs[3]])
        return torch.tensor(probs_2d, dtype=torch.float32)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        probs = self.forward(inputs)
        samples = torch.multinomial(probs, num_samples, replacement=True)
        return samples.squeeze(-1)

__all__ = ["HybridSamplerQNN"]
