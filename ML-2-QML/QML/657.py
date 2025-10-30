import pennylane as qml
import torch

class SamplerQNN:
    """Variational quantum sampler with a 2‑qubit circuit and torch‑backed PennyLane node."""
    def __init__(self,
                 n_qubits: int = 2,
                 n_layers: int = 2,
                 device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits)

        # Trainable rotation angles
        self.weights = torch.nn.Parameter(torch.randn((n_layers, n_qubits)))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode input parameters
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Return probabilities for the first qubit (2‑class output)
            return qml.probs(wires=0)

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over two outcomes."""
        return self.circuit(inputs, self.weights)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample measurement outcomes from the quantum circuit."""
        probs = self.forward(inputs)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,))

    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence KL(p || q)."""
        eps = 1e-12
        p = p + eps
        q = q + eps
        return torch.sum(p * torch.log(p / q), dim=-1)

__all__ = ["SamplerQNN"]
