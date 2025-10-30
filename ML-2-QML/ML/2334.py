"""Hybrid convolution module with classical and quantum support.

This module implements a ConvGen138 class that can operate in three modes:
- classic: uses a simple 2‑D convolution filter implemented in PyTorch.
- quantum: uses a variational quantum circuit to compute a filter response.
- hybrid: averages the outputs of the classical and quantum filters.

Additional utilities for generating random classical networks, feedforward propagation,
state fidelity, and fidelity‑based adjacency graphs are provided to mirror the
GraphQNN seed.  All functions are fully type‑annotated and importable.
"""

from __future__ import annotations

import math
import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import torch
from torch import nn
import numpy as np
import networkx as nx
import qiskit

# --------------------------------------------------------------------------- #
# Classical convolution filter
# --------------------------------------------------------------------------- #
class _ClassicConv(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        return torch.sigmoid(logits - self.threshold).mean()

# --------------------------------------------------------------------------- #
# Quantum variational filter (placeholder)
# --------------------------------------------------------------------------- #
class _QuantumConvFilter:
    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 0.0) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.circuit = self._build_circuit()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(qiskit.circuit.Parameter(f"theta{i}"), i)
        qc.barrier()
        qc += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        flat = data.view(-1).float()
        bind = {f"theta{i}": math.pi if val > self.threshold else 0 for i, val in enumerate(flat)}
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = 0
        for key, val in counts.items():
            ones = key.count("1")
            total += ones * val
        return torch.tensor(total / (self.shots * self.n_qubits))

# --------------------------------------------------------------------------- #
# Hybrid convolution filter
# --------------------------------------------------------------------------- #
class ConvGen138:
    """Hybrid convolution filter with optional quantum support.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel.
    threshold : float
        Threshold to binarize input data before feeding into the quantum circuit.
    mode : str
        One of ``'classic'``, ``'quantum'`` or ``'hybrid'``.
    shots : int
        Number of shots for the quantum simulation (ignored in classic mode).
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 mode: str = "classic",
                 shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.mode = mode
        self.shots = shots
        self.classic = _ClassicConv(kernel_size, threshold)
        self.quantum = _QuantumConvFilter(kernel_size, shots, threshold)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.mode == "classic":
            return self.classic(data)
        elif self.mode == "quantum":
            return self.quantum.forward(data)
        elif self.mode == "hybrid":
            return (self.classic(data) + self.quantum.forward(data)) / 2
        else:
            raise ValueError(f"Unknown mode {self.mode}")

# --------------------------------------------------------------------------- #
# Graph utilities (classical)
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
    return float((a_norm @ b_norm).item() ** 2)

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

__all__ = [
    "ConvGen138",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
