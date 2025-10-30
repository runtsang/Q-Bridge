import pennylane as qml
import numpy as np
import torch

class SamplerQNN:
    """
    Quantum sampler neural network implemented with PennyLane.
    Provides a variational circuit that outputs a probability distribution
    over 2**wires outcomes, with full support for PyTorch autograd.
    """
    def __init__(self, wires: int = 2, n_layers: int = 2,
                 device: str = "default.qubit", seed: int | None = None) -> None:
        self.wires = wires
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=self.wires)
        self.params = torch.nn.Parameter(
            0.01 * torch.randn((self.n_layers, self.wires, 3), requires_grad=True)
        )
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Amplitude encoding of the input
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for w in range(self.wires):
                    qml.Rot(*params[layer, w, :], wires=w)
                # Entangling layer
                for w in range(self.wires - 1):
                    qml.CNOT(wires=[w, w + 1])
            # Return probabilities for all basis states
            return qml.probs(wires=list(range(self.wires)))
        self.qnode = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability distribution for each sample in the batch.
        inputs: shape (batch, wires)
        Returns: shape (batch, 2**wires)
        """
        return self.qnode(inputs, self.params)

    def sample(self, inputs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample measurement outcomes from the quantum circuit.
        Returns integer labels in [0, 2**wires - 1].
        """
        probs = self.forward(inputs)  # (batch, 2**wires)
        probs_np = probs.detach().cpu().numpy()
        samples = [np.random.choice(len(p), size=n_samples, p=p) for p in probs_np]
        return torch.tensor(samples, dtype=torch.long)

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss between predicted probabilities and target labels.
        """
        probs = self.forward(inputs)
        log_probs = torch.log(probs + 1e-12)
        return torch.nn.functional.nll_loss(log_probs, targets)

__all__ = ["SamplerQNN"]
