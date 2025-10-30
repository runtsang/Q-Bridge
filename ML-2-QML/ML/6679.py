"""HybridGraphQuantumNet – Classical component of the unified binary‑classification pipeline.

This module defines a single PyTorch model that merges a convolutional backbone,
a two‑qubit hybrid head, and a fidelity‑based graph regularisation.  The
class name `HybridGraphQuantumNet` is shared with its quantum counterpart
in the QML module, enabling seamless hybrid training.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from qiskit import Aer


# --------------------------------------------------------------------------- #
# 1. Random utilities (from GraphQNN)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random dense layer weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic training pairs for a linear target."""
    data: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        data.append((features, target))
    return data


# --------------------------------------------------------------------------- #
# 2. Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Parameterized two‑qubit circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the parametrised circuit for the provided angles."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


# --------------------------------------------------------------------------- #
# 3. Hybrid layer
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs to be a 1‑D tensor of parameters
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        thetas = squeezed.numpy()
        expectation_z = self.quantum_circuit.run(thetas)
        return torch.tensor([expectation_z[0]])


# --------------------------------------------------------------------------- #
# 4. Graph utilities (from GraphQNN)
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared overlap between two unit‑norm tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: list[torch.Tensor], threshold: float,
    *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
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
# 5. HybridGraphQuantumNet
# --------------------------------------------------------------------------- #
class HybridGraphQuantumNet(nn.Module):
    """CNN + hybrid quantum head + graph‑regularised output."""

    def __init__(self, n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Hybrid quantum head
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class probabilities and a graph of quantum states."""
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Hybrid quantum head
        prob = self.hybrid(x)
        # Build a graph from the hybrid output as a single state
        graph = fidelity_adjacency([prob], threshold=0.9)
        # Return probabilities and graph
        return torch.cat((prob, 1 - prob), dim=-1), graph


__all__ = ["HybridGraphQuantumNet", "QuantumCircuit", "Hybrid", "state_fidelity", "fidelity_adjacency"]
