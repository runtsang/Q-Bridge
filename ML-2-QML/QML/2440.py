"""
Quantum hybrid regression model using Pennylane.

The model encodes input states via amplitude encoding, applies a
graph‑structured entangling layer, and a variational block.  The
classical regression head maps the measurement expectations to a
scalar output.  Fidelity‑based adjacency can be used to augment the
classical features if desired.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import networkx as nx
from typing import Sequence, Iterable

# Import utilities from the anchor GraphQNN module
from GraphQNN import fidelity_adjacency, state_fidelity

# Import the dataset generator from the quantum regression seed
from QuantumRegression import generate_superposition_data


class HybridGraphQNNRegressor(qml.nn.Module):
    """
    Quantum‑classical regression model that mirrors the classical
    HybridGraphQNNRegressor but replaces the graph neural network
    with a variational quantum circuit whose entangling pattern
    is derived from the same adjacency graph used in the classical
    counterpart.
    """

    def __init__(
        self,
        num_wires: int,
        qnn_arch: Sequence[int],
        hidden_dim: int = 32,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_wires = num_wires
        self.qnn_arch = list(qnn_arch)
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        # Quantum device with batch support
        self.dev = qml.device("default.qubit", wires=num_wires, shots=0)

        # Build a simple graph adjacency for entanglement
        self.graph = nx.path_graph(num_wires)

        # Variational parameters
        self.var_params = nn.Parameter(torch.randn(2 * num_wires, requires_grad=True))

        # Classical regression head
        self.head = nn.Linear(num_wires, 1, device=self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Create the qnode
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")

    def circuit(self, inputs: torch.Tensor, var_params: torch.Tensor) -> torch.Tensor:
        """
        Pennylane qnode that encodes *inputs* via amplitude encoding,
        applies entanglement according to *self.graph*, and a variational
        block of Ry rotations.
        """
        # Amplitude encoding
        qml.StatePrep(inputs, wires=range(self.num_wires))

        # Entanglement pattern from the graph
        for edge in self.graph.edges():
            qml.CNOT(wires=edge)

        # Variational layer
        for i in range(self.num_wires):
            qml.RY(var_params[2 * i], wires=i)
            qml.RZ(var_params[2 * i + 1], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: run the quantum circuit for each batch sample
        and feed the measurement outcomes to the classical head.
        """
        # Ensure device alignment
        x = x.to(self.device)
        # Run the qnode in batch mode
        q_out = self.qnode(x, self.var_params)
        # Classical head
        return self.head(q_out).squeeze(-1)

    def train_on_batch(self, batch: dict[str, torch.Tensor]) -> float:
        """
        One training step on a single batch.
        """
        self.train()
        self.optimizer.zero_grad()
        preds = self.forward(batch["states"])
        loss = self.loss_fn(preds, batch["target"])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, loader: Iterable[dict[str, torch.Tensor]]) -> float:
        """
        Compute mean squared error over a data loader.
        """
        self.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                preds = self.forward(batch["states"])
                loss = self.loss_fn(preds, batch["target"])
                total_loss += loss.item() * batch["states"].size(0)
                count += batch["states"].size(0)
        return total_loss / count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(wires={self.num_wires}, "
            f"arch={self.qnn_arch}, hidden_dim={self.hidden_dim})"
        )
