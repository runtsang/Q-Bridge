import pennylane as qml
import numpy as np
import torch
from typing import Sequence

class SamplerQNN:
    """
    Quantum sampler network built on a parameterised ansatz.
    Supports automatic differentiation via PennyLane's autograd.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        device: str = "default.qubit",
        use_tape: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits in the circuit.
        depth : int
            Number of entangling layers.
        device : str
            PennyLane device name.
        use_tape : bool
            Whether to precompile the circuit into a QNode tape for speed.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device(device, wires=num_qubits)
        self.wires = list(range(num_qubits))
        # Trainable parameters: one RY per qubit per layer
        self.weights = torch.nn.Parameter(
            0.01 * torch.randn(num_qubits * depth, requires_grad=True)
        )
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Variational circuit producing a probability distribution over
        computational basis states via a statevector measurement.
        """
        idx = 0
        for _ in range(self.depth):
            for w in range(self.num_qubits):
                qml.RY(weights[idx], wires=w)
                idx += 1
            # Entangle qubits in a ring
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])
        # Return full statevector
        return qml.state()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability distribution for a batch of inputs.
        The `inputs` tensor is ignored in this simple example but kept for API
        compatibility with hybrid models.
        """
        probs = []
        for _ in inputs:
            sv = self.qnode(self.weights)
            probs.append(np.abs(sv) ** 2)
        return torch.tensor(probs)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the quantum probability distribution.
        Returns a tensor of shape (num_samples, num_qubits).
        """
        probs = self.forward(torch.zeros(1))  # dummy input
        dist = torch.distributions.Categorical(probs[0])
        return dist.sample((num_samples,))

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        """
        Simple cross‑entropy loss against a target distribution.
        """
        probs = self.forward(torch.zeros(1))
        return torch.nn.functional.nll_loss(
            torch.log(probs[0] + 1e-8), target
        )
