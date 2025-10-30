from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Sequence, Tuple, Callable

# ------------------------------------------------------------------
# Classical kernel utilities
# ------------------------------------------------------------------
class RBFKernel(nn.Module):
    """Radial‑basis function kernel used for graph construction."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# ------------------------------------------------------------------
# Quanvolution primitives
# ------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """2×2 patch extraction followed by a shallow conv layer."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier built on top of the quanvolution filter."""
    def __init__(self, num_classes: int = 10, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels)
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# ------------------------------------------------------------------
# Fast estimator for deterministic models
# ------------------------------------------------------------------
class FastEstimator:
    """Batch evaluator that accepts a list of scalar observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.model(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().item()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results

# ------------------------------------------------------------------
# Core hybrid graph‑neural‑network
# ------------------------------------------------------------------
class GraphQNNHybrid(nn.Module):
    """
    Classical graph neural network that:
      * builds a random feed‑forward network,
      * propagates samples through the network,
      * constructs a graph via an RBF kernel,
      * offers a fast estimator for batch evaluation.
    """
    def __init__(self,
                 arch: Sequence[int],
                 gamma: float = 1.0,
                 random_seed: int | None = None) -> None:
        super().__init__()
        self.arch = list(arch)
        self.gamma = gamma
        self.random_seed = random_seed
        self.weights: List[torch.Tensor] = []
        self._build_random_network()
        self.kernel = RBFKernel(gamma)

    # ------------------------------------------------------------------
    # Random network construction
    # ------------------------------------------------------------------
    def _build_random_network(self) -> None:
        rng = torch.Generator()
        if self.random_seed is not None:
            rng.manual_seed(self.random_seed)
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            w = torch.randn(out_f, in_f, generator=rng, dtype=torch.float32)
            self.weights.append(w)

    @staticmethod
    def random_network(arch: Sequence[int], samples: int, random_seed: int | None = None
                      ) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Convenience helper that returns architecture, weights, training data and target weight."""
        rng = torch.Generator()
        if random_seed is not None:
            rng.manual_seed(random_seed)
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(torch.randn(out_f, in_f, generator=rng, dtype=torch.float32))
        target_weight = weights[-1]
        # Generate synthetic data for the last layer
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            target = target_weight @ features
            dataset.append((features, target))
        return list(arch), weights, dataset, target_weight

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def feedforward(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations: List[torch.Tensor] = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        return activations

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def fidelity_adjacency(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from RBF similarities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, xi), (j, xj) in itertools.combinations(enumerate(states), 2):
            fid = self.kernel(xi, xj).item()
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets)

    # ------------------------------------------------------------------
    # Synthetic training data generator
    # ------------------------------------------------------------------
    def random_training_data(self, target_weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            target = target_weight @ features
            data.append((features, target))
        return data

__all__ = [
    "RBFKernel",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "FastEstimator",
    "GraphQNNHybrid",
]
