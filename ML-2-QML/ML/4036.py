"""Hybrid Classical Sampler QNN combining classical encoding, quantum simulation, and fully connected layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    A hybrid classical neural network that simulates a parameterised 2‑qubit quantum sampler
    and feeds the resulting expectation values into a fully‑connected classifier.
    The architecture mirrors the original SamplerQNN (softmax over 2‑D input) and the
    fully‑connected layer (expectation of a single qubit), but replaces the quantum
    backend with a differentiable numpy‑style simulation.
    """

    def __init__(self) -> None:
        super().__init__()

        # Classical encoder that produces the two “input” parameters for the quantum circuit.
        self.encoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # Learnable weight parameters for the quantum circuit (4 rotations).
        self.weights = nn.Parameter(torch.randn(4, dtype=torch.float32))

        # Final classifier that maps the quantum expectation to a scalar output.
        self.classifier = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 2) containing the classical input features.

        Returns:
            Tensor of shape (batch, 1) containing the network output.
        """
        # Encode classical input to quantum parameters
        q_params = self.encoder(x)  # (batch, 2)

        # Compute quantum expectation for each input sample
        expectations = torch.stack(
            [self._quantum_sampler(p[0], p[1]) for p in q_params], dim=0
        )  # (batch, 1)

        # Pass through final classifier
        out = self.classifier(expectations)
        return out

    def _quantum_sampler(self, theta0: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
        """
        Simulate the 2‑qubit circuit and return the expectation of Z on qubit 0.

        The circuit is identical to the QML reference:
        Ry(theta0) on qubit0 → Ry(theta1) on qubit1 → CX → Ry(w0) → Ry(w1)
        → CX → Ry(w2) → Ry(w3).  The expectation value of Z on qubit 0 is
        returned as a scalar tensor.

        Args:
            theta0: Rotation angle for qubit 0 (input).
            theta1: Rotation angle for qubit 1 (input).

        Returns:
            Tensor of shape (1,) containing the expectation value.
        """
        # Helper to construct Ry unitary
        def ry(theta: torch.Tensor) -> torch.Tensor:
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            return torch.tensor(
                [[c, -s], [s, c]], dtype=torch.complex64, device=theta.device
            )

        # Basis state |00>
        state = torch.zeros((4, 1), dtype=torch.complex64, device=theta0.device)
        state[0, 0] = 1.0 + 0.0j

        # Apply Ry on qubit0
        U0 = torch.kron(ry(theta0), torch.eye(2, dtype=torch.complex64, device=theta0.device))
        state = U0 @ state

        # Apply Ry on qubit1
        U1 = torch.kron(torch.eye(2, dtype=torch.complex64, device=theta0.device), ry(theta1))
        state = U1 @ state

        # Controlled‑X gate
        cx = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=torch.complex64,
            device=theta0.device,
        )
        state = cx @ state

        # Apply weight rotations
        for w in self.weights:
            Uw = torch.kron(ry(w), torch.eye(2, dtype=torch.complex64, device=theta0.device))
            state = Uw @ state

        # Apply second CX
        state = cx @ state

        # Final weight rotations on qubit1
        for w in self.weights[2:]:
            Uw = torch.kron(torch.eye(2, dtype=torch.complex64, device=theta0.device), ry(w))
            state = Uw @ state

        # Expectation of Z on qubit 0: 〈ψ| Z⊗I |ψ〉
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=theta0.device)
        Z0 = torch.kron(Z, torch.eye(2, dtype=torch.complex64, device=theta0.device))
        expectation = torch.conj(state.t()) @ Z0 @ state
        return expectation.squeeze()

__all__ = ["HybridSamplerQNN"]
