"""HybridBinaryClassifier: quantum‑enhanced binary classifier.

The implementation mirrors the classical version but replaces the final
dense head with a parameterised quantum circuit.  A graph derived from
feature fidelities determines the entanglement pattern, enabling
relational reasoning at the quantum level.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
import qiskit
from qiskit import assemble, transpile

# --- Utility functions ---------------------------------------------------------

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- Quantum circuit wrapper ----------------------------------------------------

class QuantumCircuitWrapper:
    """Parameterized circuit that encodes each sample as a rotation
    and entangles qubits according to a graph adjacency.
    """

    def __init__(self,
                 n_qubits: int,
                 backend,
                 shots: int,
                 shift: float) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.adjacency: nx.Graph | None = None

    def set_adjacency(self, adjacency: nx.Graph) -> None:
        self.adjacency = adjacency

    def run(self, thetas: np.ndarray, adjacency: nx.Graph) -> np.ndarray:
        circuit = qiskit.QuantumCircuit(self.n_qubits)
        # encode features as RY rotations
        for q in range(self.n_qubits):
            circuit.ry(thetas[q], q)
        # entangle according to adjacency
        for i, j in adjacency.edges():
            circuit.cx(i, j)
        circuit.measure_all()
        compiled = transpile(circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        expectations = np.zeros(self.n_qubits)
        for outcome, count in counts.items():
            prob = count / self.shots
            for q in range(self.n_qubits):
                bit = int(outcome[self.n_qubits - 1 - q])  # little‑endian
                expectations[q] += (1 - 2 * bit) * prob
        return expectations

# --- Hybrid autograd function ----------------------------------------------------

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        theta = inputs.detach().cpu().numpy().squeeze()
        expectations = ctx.circuit.run(theta, ctx.circuit.adjacency)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        theta = inputs.detach().cpu().numpy().squeeze()
        gradients = np.zeros_like(theta)
        for idx in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[idx] += shift
            theta_minus[idx] -= shift
            exp_plus = ctx.circuit.run(theta_plus, ctx.circuit.adjacency)
            exp_minus = ctx.circuit.run(theta_minus, ctx.circuit.adjacency)
            gradients[idx] = (exp_plus[idx] - exp_minus[idx]) / (2 * shift)
        grad_inputs = torch.tensor(gradients, dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output, None, None

# --- Hybrid binary classifier ----------------------------------------------------

class HybridBinaryClassifier(nn.Module):
    """CNN + quantum expectation head driven by graph‑based entanglement."""

    def __init__(self,
                 in_channels: int = 3,
                 graph_threshold: float = 0.8,
                 shift: float = np.pi / 2,
                 shots: int = 100) -> None:
        super().__init__()
        self.graph_threshold = graph_threshold
        self.shift = shift
        self.shots = shots

        # CNN backbone
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_circuit: QuantumCircuitWrapper | None = None

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)  # (batch,)

        batch_size = x.size(0)

        # Build adjacency across the batch
        adjacency = fidelity_adjacency(
            [x[i].unsqueeze(0) for i in range(batch_size)],
            self.graph_threshold,
        )

        # Quantum circuit for the current batch
        self.quantum_circuit = QuantumCircuitWrapper(
            n_qubits=batch_size,
            backend=self.backend,
            shots=self.shots,
            shift=self.shift,
        )
        self.quantum_circuit.set_adjacency(adjacency)

        # Quantum expectation
        quantum_output = HybridFunction.apply(x, self.quantum_circuit, self.shift)
        return self.sigmoid(quantum_output)

__all__ = ["HybridBinaryClassifier"]
