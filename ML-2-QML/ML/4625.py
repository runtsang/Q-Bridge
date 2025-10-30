"""Hybrid convolutional filter that can operate in either classical or quantum mode.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

import itertools
import numpy as np
import random
import torch
from torch import nn
from typing import Iterable, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------
# Classical convolution component
# ------------------------------------------------------------------
class _ClassicConv(nn.Module):
    """Deterministic 2‑D convolution mirroring the original Conv filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, weight_init: str = "xavier") -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        if weight_init == "xavier":
            nn.init.xavier_uniform_(self.conv.weight)
        elif weight_init == "zeros":
            nn.init.constant_(self.conv.weight, 0.0)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Expect data shape [H, W] where H and W equal kernel_size
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

# ------------------------------------------------------------------
# Quantum filter simulation (classical approximation)
# ------------------------------------------------------------------
class _QuantumFilterSim:
    """Lightweight classical simulation of a variational quantum filter."""
    def __init__(self, kernel_size: int, shots: int = 100, threshold: float = 127.0, seed: Optional[int] = None):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Map kernel values to rotation angles."""
        return np.where(data > self.threshold, np.pi, 0.0)

    def run(self, data: np.ndarray) -> float:
        """Simulate measurement statistics."""
        data = data.reshape(1, self.n_qubits)
        angles = self.encode(data[0])
        # Simulate a random circuit by adding Gaussian noise to angles
        noisy_angles = angles + self.rng.normal(scale=0.1, size=angles.shape)
        probs = np.abs(np.sin(noisy_angles / 2)) ** 2  # probability of |1> for each qubit
        counts = np.sum(probs) * self.shots
        return counts / (self.shots * self.n_qubits)

# ------------------------------------------------------------------
# Hybrid Conv layer
# ------------------------------------------------------------------
class ConvQuantumHybrid(nn.Module):
    """Drop‑in replacement for Conv that can operate in classical or quantum mode."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_quantum: bool = False,
        quantum_shots: int = 100,
        quantum_threshold: float = 127.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.classic = _ClassicConv(kernel_size, threshold)
        if use_quantum:
            self.quantum = _QuantumFilterSim(kernel_size, quantum_shots, quantum_threshold, seed)
        else:
            self.quantum = None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run either the classical conv or the quantum simulation."""
        if self.use_quantum and self.quantum is not None:
            arr = data.detach().cpu().numpy()
            prob = self.quantum.run(arr)
            return torch.tensor(prob, dtype=torch.float32)
        else:
            return self.classic(data)

    def encode(self, data: torch.Tensor) -> np.ndarray:
        """Utility that mirrors the quantum encoding routine."""
        arr = data.detach().cpu().numpy()
        return np.where(arr > self.threshold, np.pi, 0.0)

# ------------------------------------------------------------------
# Classical classifier builder (inspired by reference 2)
# ------------------------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Construct a simple feed‑forward network with metadata mirroring the quantum interface."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# ------------------------------------------------------------------
# Graph‑based utilities (inspired by reference 3)
# ------------------------------------------------------------------
def random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix with standard normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random MLP and a training set for its final weight matrix."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Run a forward pass through the MLP for a collection of samples."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations: List[torch.Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared inner product of two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> "networkx.Graph":
    """Construct a graph where edges are weighted by fidelity thresholds."""
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "ConvQuantumHybrid",
    "build_classifier_circuit",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
