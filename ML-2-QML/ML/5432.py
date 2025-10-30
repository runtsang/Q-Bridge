"""Combined classical graph neural network and estimator utilities.

The class ``GraphQNNGen325`` merges graph‑based propagation, fidelity‑based adjacency,
classical self‑attention and classification primitives, and a lightweight
estimator that can add Gaussian shot noise.  It provides a static
interface that mirrors the quantum counterpart defined in the sibling
module, making it trivial to swap between classical and quantum
back‑ends in downstream experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

__all__ = ["GraphQNNGen325"]


class GraphQNNGen325:
    """Unified classical interface for graph‑based neural nets and estimators."""

    # ------------------------------------------------------------------
    #  Graph‑based network construction
    # ------------------------------------------------------------------
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random weight matrix."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate a toy dataset by applying a fixed linear transformation."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(weight.size(1), dtype=torch.float32)
            y = weight @ x
            dataset.append((x, y))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create a layered linear network together with synthetic data."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNNGen325._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNNGen325.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Propagate all samples through the linear stack."""
        stored: List[List[Tensor]] = []
        for x, _ in samples:
            activations = [x]
            current = x
            for w in weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two vectors."""
        anorm = a / (torch.norm(a) + 1e-12)
        bnorm = b / (torch.norm(b) + 1e-12)
        return float((anorm @ bnorm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state overlaps."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen325.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    #  Classical classifier factory
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Sequence[int], List[int], Sequence[int]]:
        """Return a feed‑forward classifier and metadata."""
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            lin = nn.Linear(in_dim, num_features)
            layers.extend([lin, nn.ReLU()])
            weight_sizes.append(lin.weight.numel() + lin.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    # ------------------------------------------------------------------
    #  Classical self‑attention
    # ------------------------------------------------------------------
    @staticmethod
    def SelfAttention(embed_dim: int = 4):
        """Return a simple attention wrapper mimicking the quantum block."""

        class ClassicalSelfAttention:
            def __init__(self, embed_dim: int = embed_dim):
                self.embed_dim = embed_dim

            def run(
                self,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                inputs: np.ndarray,
            ) -> np.ndarray:
                q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
                k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
                v = torch.as_tensor(inputs, dtype=torch.float32)
                scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
                return (scores @ v).numpy()

        return ClassicalSelfAttention()

    # ------------------------------------------------------------------
    #  Fast estimators
    # ------------------------------------------------------------------
    class FastBaseEstimator:
        """Evaluate a torch model on batched inputs and observables."""

        def __init__(self, model: nn.Module) -> None:
            self.model = model

        def evaluate(
            self,
            observables: Iterable[ScalarObservable],
            parameter_sets: Sequence[Sequence[float]],
        ) -> List[List[float]]:
            observables = list(observables) or [lambda out: out.mean(dim=-1)]
            results: List[List[float]] = []
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    batch = torch.as_tensor(params, dtype=torch.float32)
                    if batch.ndim == 1:
                        batch = batch.unsqueeze(0)
                    out = self.model(batch)
                    row: List[float] = []
                    for obs in observables:
                        val = obs(out)
                        if isinstance(val, torch.Tensor):
                            val = float(val.mean().cpu())
                        else:
                            val = float(val)
                        row.append(val)
                    results.append(row)
            return results

    class FastEstimator(FastBaseEstimator):
        """Add Gaussian shot noise to the deterministic estimator."""

        def evaluate(
            self,
            observables: Iterable[ScalarObservable],
            parameter_sets: Sequence[Sequence[float]],
            *,
            shots: int | None = None,
            seed: int | None = None,
        ) -> List[List[float]]:
            raw = super().evaluate(observables, parameter_sets)
            if shots is None:
                return raw
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in raw:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
