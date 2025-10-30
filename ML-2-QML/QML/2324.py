"""GraphQNNHybrid – quantum‑classical graph neural network.

This module implements a graph neural network that uses a quantum
circuit as the final read‑out.  The quantum part is a variational
circuit that takes the node feature vector as rotation angles and
returns the expectation of the Pauli‑Z operator on the first qubit.
The circuit is differentiable via a custom autograd function that
uses the parameter‑shift rule.
"""

from __future__ import annotations

import itertools
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer, assemble, transpile
from qiskit.circuit import Parameter

# ---------- Quantum utilities ----------
class QuantumCircuit:
    """Wrapper around a parametrised circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = Parameter("theta")
        # Simple circuit: H on all qubits, then Ry(theta) on each qubit, then measure
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        for q in all_qubits:
            self._circuit.ry(self.theta, q)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the parametrised circuit for the provided angles."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # expectation of Z on first qubit
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k[0]) for k in count_dict.keys()])  # first qubit bit
            probs = counts / self.shots
            return np.sum((2 * states - 1) * probs)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.detach().cpu().numpy()) * ctx.shift
        grads = []
        for idx, value in enumerate(inputs.detach().cpu().numpy()):
            right = ctx.circuit.run([value + shift[idx]])[0]
            left = ctx.circuit.run([value - shift[idx]])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=inputs.dtype, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# ---------- Graph Neural Network ----------
class GraphQNNHybrid(nn.Module):
    """Graph neural network with a quantum read‑out head.

    Parameters
    ----------
    arch : Sequence[int]
        Node feature dimensionality for each layer.
    qnn_arch : Sequence[int]
        Architecture for the quantum circuit.
    backend : str or qiskit backend
        Backend used by the quantum circuit.
    shots : int
        Number of shots for the quantum simulation.
    shift : float
        Shift value for parameter‑shift gradient estimation.
    """

    def __init__(self, arch: Sequence[int], qnn_arch: Sequence[int],
                 backend=None, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.arch = tuple(arch)
        self.layers: nn.ModuleList = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])])
        if backend is None:
            backend = Aer.get_backend("aer_simulator")
        self.head = Hybrid(arch[-1], backend, shots, shift)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GNN with a quantum read‑out."""
        for layer in self.layers:
            agg = adjacency @ x
            x = F.relu(layer(agg))
        return self.head(x).squeeze(-1)

# ---------- Utilities (same as ML seed) ----------
def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[Tuple[int,...], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random feed‑forward network and training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
    target_weight = weights[-1]
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1), dtype=torch.float32)
        target = target_weight @ features
        dataset.append((features, target))
    return tuple(qnn_arch), weights, dataset, target_weight

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return random_network([weight.size(0), weight.size(1)], samples)[2]

def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
