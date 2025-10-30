"""Hybrid classical‑quantum graph neural network module.

This module extends the original GraphQNN seed by adding a
`GraphQNN__gen151` class that combines a classical feed‑forward
network with a Pennylane variational circuit.  The quantum part
receives the classical activations as parameters and returns the
expectation value of a Pauli‑Z observable.  The class can be
trained purely classically or in a hybrid fashion using a
custom loss that compares the quantum output to a target state.
The original fidelity‑based utilities are preserved for
compatibility with existing pipelines.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import pennylane as qml

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Helper functions (kept from the seed)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix (out_features x in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate random training data using the target weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network and its training data."""
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Classical feed‑forward through a list of weight matrices."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two classical vectors treated as quantum states."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from classical state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Hybrid GraphQNN class
# --------------------------------------------------------------------------- #
class GraphQNN__gen151(torch.nn.Module):
    """Hybrid classical‑quantum graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes for the classical part.  The last layer size
        determines the dimensionality of the classical activation
        that is fed to the quantum circuit.
    num_qubits : int, optional
        Number of qubits used in the variational circuit.
    device : str, optional
        Torch device for the classical weights.
    """

    def __init__(self, arch: Sequence[int], num_qubits: int = 4, device: str = "cpu"):
        super().__init__()
        self.arch = list(arch)
        self.num_qubits = num_qubits
        self.device = device

        # Classical layers
        self.classical_layers = torch.nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.classical_layers.append(
                torch.nn.Linear(in_f, out_f, bias=True)
            )

        # Quantum parameters
        self.quantum_params = torch.nn.Parameter(
            torch.randn(num_qubits, dtype=torch.float32, device=self.device)
        )

        # Pennylane device and QNode
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def _quantum_circuit(x: torch.Tensor, params: torch.Tensor):
            """Variational circuit that receives classical activation `x`
            as a list of parameters for rotation gates."""
            # Encode classical activation as rotation angles
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Variational layer
            for i in range(self.num_qubits):
                qml.RZ(params[i], wires=i)
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of Pauli‑Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self._quantum_circuit = _quantum_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classical layers followed by quantum layer."""
        # Classical feed‑forward
        for layer in self.classical_layers:
            x = torch.tanh(layer(x))
        # Quantum evaluation
        if x.shape[-1]!= self.num_qubits:
            if x.shape[-1] < self.num_qubits:
                pad = torch.zeros(self.num_qubits - x.shape[-1], device=x.device)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., : self.num_qubits]
        quantum_out = self._quantum_circuit(x, self.quantum_params)
        return quantum_out

    # --------------------------------------------------------------------- #
    #  Training helpers
    # --------------------------------------------------------------------- #
    def train_classical(
        self,
        data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> None:
        """Train only the classical weights using MSE loss."""
        optimizer = torch.optim.Adam(self.classical_layers.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                out = self.forward(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

    def train_hybrid(
        self,
        data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> None:
        """Train both classical and quantum parameters using MSE loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                out = self.forward(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

    def evaluate_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the quantum output for a given input."""
        return self._quantum_circuit(x, self.quantum_params)

# --------------------------------------------------------------------------- #
#  Exports
# --------------------------------------------------------------------------- #
__all__ = [
    "GraphQNN__gen151",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
