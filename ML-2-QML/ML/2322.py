from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Callable, Tuple

import torch
import torch.nn as nn
import numpy as np
import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class GraphQNN(nn.Module):
    """Classical graph neural network with PyTorch backend.

    The network architecture is specified by a sequence of layer sizes.
    Each layer consists of a linear transform followed by a tanh activation.
    """

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate random weights and synthetic training data."""
        # Build random weights
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            w = torch.randn(out_f, in_f, dtype=torch.float32)
            weights.append(w)
        target_weight = weights[-1]
        # Synthetic dataset using the last layer's weight
        training_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1))
            target = target_weight @ features
            training_data.append((features, target))
        return list(arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        arch: Sequence[int],
        weights: Sequence[torch.Tensor],
        samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """Return activations for each sample."""
        activations: List[List[torch.Tensor]] = []
        for features, _ in samples:
            act = [features]
            current = features
            for w in weights:
                current = torch.tanh(w @ current)
                act.append(current)
            activations.append(act)
        return activations

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared overlap of two normalized vectors."""
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float((a_n @ b_n).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate observables on batches of input parameters."""
        if not observables:
            observables = [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self.net(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().item())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to deterministic outputs."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["GraphQNN"]
