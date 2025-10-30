"""ConvGen module providing a hybrid classical/quantum convolutional filter and graph utilities.

The module offers:
  * ConvGen: hybrid filter with classic PyTorch or Qiskit variational circuit.
  * GraphConv: graph‑convolutional layer built from fidelity‑based adjacency.
  * Helper functions for random networks, training data, feedforward, state fidelity, and adjacency construction.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Literal, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

# --------------------------------------------------------------------------- #
# 1. Hybrid convolution filter
# --------------------------------------------------------------------------- #

class ConvGen(nn.Module):
    """Hybrid convolution filter that can run in classic or quantum mode.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the 2‑D kernel.
    threshold : float, default 0.0
        Threshold used in the sigmoid activation (classic) or in the
        parameter binding (quantum).
    mode : {'classic', 'quantum'}, default 'classic'
        Operation mode.
    shots : int, default 100
        Number of shots for the quantum backend.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        mode: Literal["classic", "quantum"] = "classic",
        shots: int = 100,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.mode = mode
        self.shots = shots

        if self.mode == "classic":
            # Classic 1‑channel 1‑output convolution
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        else:
            # Quantum circuit
            self.n_qubits = kernel_size ** 2
            self.circuit = self._build_quantum_circuit(self.n_qubits)
            self.backend = Aer.get_backend("qasm_simulator")

    def _build_quantum_circuit(self, n_qubits: int) -> qiskit.QuantumCircuit:
        """Create a quantum circuit that applies random rotations and entanglement."""
        circ = qiskit.QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta_{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            circ.rx(theta[i], i)
        circ.barrier()
        # Simple entangling layer
        for i in range(0, n_qubits - 1, 2):
            circ.cx(i, i + 1)
        circ.measure_all()
        return circ

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """Run the filter on a 2‑D kernel.

        Parameters
        ----------
        data : array‑like of shape (kernel_size, kernel_size)
            Input kernel.

        Returns
        -------
        float
            Mean activation (classic) or mean measurement probability (quantum).
        """
        if self.mode == "classic":
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()
        else:
            # Quantum evaluation
            data_flat = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data_flat:
                bind = {f"theta_{i}": np.pi if val > self.threshold else 0
                        for i, val in enumerate(dat)}
                param_binds.append(bind)
            job = execute(self.circuit,
                          self.backend,
                          shots=self.shots,
                          parameter_binds=param_binds)
            result = job.result().get_counts(self.circuit)
            total = 0
            for bitstring, count in result.items():
                ones = sum(int(b) for b in bitstring)
                total += ones * count
            return total / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# 2. Graph utilities
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
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
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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
# 3. Graph‑convolutional layer
# --------------------------------------------------------------------------- #

class GraphConv(nn.Module):
    """Graph‑convolutional layer that uses a fidelity‑based adjacency matrix.

    Parameters
    ----------
    in_features : int
        Size of each input node feature vector.
    out_features : int
        Size of each output node feature vector.
    threshold : float, default 0.8
        Fidelity threshold for adjacency edge creation.
    """

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, states: Sequence[torch.Tensor]) -> torch.Tensor:
        """Apply graph convolution.

        Parameters
        ----------
        node_features : Tensor of shape (num_nodes, in_features)
        states : Sequence[Tensor] of length num_nodes, used to compute adjacency.

        Returns
        -------
        Tensor of shape (num_nodes, out_features)
        """
        graph = fidelity_adjacency(states, self.threshold)
        adj = nx.to_numpy_array(graph, nodelist=range(len(states)))
        deg = np.diag(1 / (adj.sum(axis=1) + 1e-6))
        agg = torch.from_numpy(adj @ deg).float() @ node_features
        return self.linear(agg)

__all__ = [
    "ConvGen",
    "GraphConv",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
