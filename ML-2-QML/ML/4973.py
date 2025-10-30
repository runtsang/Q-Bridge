"""HybridSelfAttention: a unified classical and quantum self‑attention framework.

The module exposes a single class that implements:
* PyTorch‑based self‑attention with optional 2‑D convolution filtering.
* Fidelity‑based graph construction from arbitrary state vectors.
* Lightweight batch evaluation using user supplied scalar observables.
* Random network and training data generators for quick prototyping.

The API matches the original ``SelfAttention`` interface while extending it
with richer functionality from the graph‑QNN and estimator utilities.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from typing import Callable, Iterable, List, Sequence, Tuple

ScalarObservable = Callable[[np.ndarray], float | np.ndarray]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence to a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class _ConvFilter(nn.Module):
    """Simple 2‑D convolution filter implemented with PyTorch."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class HybridSelfAttention:
    """PyTorch implementation of a self‑attention block with optional convolution."""

    def __init__(self, embed_dim: int, kernel_size: int = 2, conv_threshold: float = 0.0) -> None:
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv_filter = _ConvFilter(kernel_size, conv_threshold) if kernel_size > 0 else None

    # ------------------------------------------------------------------
    # Classical self‑attention
    # ------------------------------------------------------------------
    def run_self_attention(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """Compute self‑attention output for a single example."""
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)

        query = inp @ rot
        key = inp @ ent
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        value = inp
        return (scores @ value).numpy()

    def apply_convolution(self, data: np.ndarray) -> float:
        """Apply the optional 2‑D convolution filter."""
        if self.conv_filter is None:
            raise RuntimeError("No convolution filter configured.")
        return self.conv_filter(data)

    # ------------------------------------------------------------------
    # Graph utilities (from GraphQNN)
    # ------------------------------------------------------------------
    @staticmethod
    def _state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a_norm, b_norm) ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Data generators (from GraphQNN)
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = HybridSelfAttention.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    # ------------------------------------------------------------------
    # Evaluation (from FastBaseEstimator)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> List[List[float]]:
        """
        Evaluate scalar observables on the outputs of ``run_self_attention``.
        ``parameter_sets`` should be an iterable of tuples:
        (inputs, rotation_params, entangle_params).
        """
        observables = list(observables) or [lambda out: np.mean(out)]
        results: List[List[float]] = []

        for inp, rot, ent in parameter_sets:
            out = self.run_self_attention(inp, rot, ent)
            row = [float(obs(out)) for obs in observables]
            results.append(row)

        return results


__all__ = ["HybridSelfAttention"]
