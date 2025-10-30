"""Quantum sampler network.

This module defines SamplerQNN as a quantum variational circuit that
produces a probability distribution over two outcomes.  The circuit
consists of input‑dependent RY rotations, a CX entangling gate and
trainable RY rotations.  The class is compatible with PyTorch
optimisers via the Pennylane torch interface.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

# 2‑qubit device, default to Aer simulator
dev = qml.device("default.qubit", wires=2, shots=1024)


class SamplerQNN(nn.Module):
    """Quantum sampler with parameter‑shift differentiable QNode.

    The network takes a 2‑dimensional classical input and returns a
    probability distribution over two outcomes.  Trainable weights
    are stored as torch parameters.
    """

    def __init__(self) -> None:
        super().__init__()

        # Trainable weight parameters (4 real numbers)
        self.weights = nn.Parameter(torch.randn(4))

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # input‑dependent rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # entanglement
            qml.CNOT(wires=[0, 1])
            # trainable rotations
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            # measurement: return probabilities of |00> and |01>
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over two outcomes.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) containing classical input
            features.  Each sample is processed independently.
        """
        # Ensure inputs are of shape (batch, 2)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        probs = self.circuit(inputs, self.weights)
        # probs shape: (..., 4). We collapse to two outcomes:
        # outcome 0: |00>, outcome 1: |01> + |10> + |11>
        outcome0 = probs[..., 0]
        outcome1 = probs[..., 1:].sum(-1)
        return torch.stack([outcome0, outcome1], dim=-1)

    def sample(self, inputs: torch.Tensor, n_samples: int = 1024) -> torch.Tensor:
        """Generate explicit samples from the quantum circuit.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) with classical inputs.
        n_samples : int
            Number of Monte‑Carlo samples to draw per input.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., n_samples, 2) containing bit‑strings
            (0 or 1) for the two qubits.
        """
        dev = qml.device("default.qubit", wires=2, shots=n_samples)

        @qml.qnode(dev, interface="torch")
        def sample_circuit(inputs: torch.Tensor, weights: torch.Tensor):
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            return qml.sample(wires=[0, 1])

        return sample_circuit(inputs, self.weights)

__all__ = ["SamplerQNN"]
