"""Hybrid fraud‑detection model using a classical network and a PennyLane quantum circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

# Import the quantum module.  In a real project this would be a separate package.
from. import FraudQuantumCircuit  # noqa: F401

@dataclass
class FraudLayerParameters:
    """Parameters for a classical layer that receives quantum expectations."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionModel(nn.Module):
    """Hybrid fraud‑detection model.

    The model consists of:
    1. A PennyLane quantum circuit that produces two expectation values.
    2. A small classical feed‑forward network that maps the quantum outputs to a
       fraud probability.
    """

    def __init__(self,
                 quantum_params: FraudLayerParameters,
                 classical_layers: Iterable[FraudLayerParameters],
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

        # Quantum circuit
        self.quantum_circuit = FraudQuantumCircuit.FraudQuantumCircuit(quantum_params)

        # Classical network
        self.classical = nn.Sequential()
        for idx, params in enumerate(classical_layers):
            self.classical.add_module(f"layer{idx}", self._make_layer(params))

        # Output layer
        self.out = nn.Linear(2, 1)

    def _make_layer(self, params: FraudLayerParameters) -> nn.Module:
        """Build a single classical layer with a linear transform and Tanh."""
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        return nn.Sequential(linear, nn.Tanh(), nn.Linear(2, 2))

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        # Quantum part – no classical input is required for the circuit.
        q_out = torch.tensor(self.quantum_circuit(), dtype=torch.float32)
        # Classical mapping
        x = self.classical(q_out)
        logits = self.out(x)
        prob = torch.sigmoid(logits / self.temperature)
        return prob.squeeze()

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(y_pred, y_true)

    def train_step(self,
                   optimizer: torch.optim.Optimizer,
                   data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Perform one epoch of training."""
        self.train()
        for x, y in data_loader:
            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()
