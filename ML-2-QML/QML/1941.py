"""Quantum sampler network using PennyLane.

The quantum version is a parameterized circuit that mirrors the
classical network's interface. It uses a 2‑qubit ansatz with alternating
single‑qubit rotations and CNOT entangling gates. The circuit is
wrapped in a PennyLane QNode that returns measurement probabilities
over the computational basis. The class exposes a `forward` method
compatible with the classical SamplerQNN for hybrid training.

Key extensions:
- Flexible number of layers and rotation types.
- Built‑in statevector simulator for exact probability evaluation.
- Optional integration with Pennylane's gradient tools for end‑to‑end
  training with PyTorch autograd.
"""

import pennylane as qml
import torch
import torch.nn as nn
from typing import Tuple


class SamplerQNN(nn.Module):
    """Quantum sampler network implemented with PennyLane."""

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits)
        # Parameter shape: (n_layers, n_qubits, 3) for (RY, RX, RZ)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor):
            # Input encoding via Ry rotations
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)
            # Variational layers
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(params[layer, qubit, 0], wires=qubit)
                    qml.RX(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)
                # Entangling CNOT ladder
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            # Measurement probabilities
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for 2‑qubit output.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch, n_qubits). Values are expected
            in the range [-π, π] to serve as Ry angles.
        """
        batch_probs = []
        for inp in x:
            probs = self.circuit(inp, self.params)  # shape (2**n_qubits,)
            # For a 2‑class output we collapse the full distribution to two
            # probabilities. Here we simply take the first two basis states.
            probs_2 = probs[:2]
            probs_2 = probs_2 / probs_2.sum()
            batch_probs.append(probs_2)
        return torch.stack(batch_probs, dim=0)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw samples from the quantum categorical distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch, n_qubits).
        n_samples : int
            Number of samples per input.
        """
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((n_samples,)).permute(1, 0, 2)  # shape (batch, n_samples, 2)


__all__ = ["SamplerQNN"]
