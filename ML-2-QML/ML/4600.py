"""Hybrid classical estimator with optional shot‑noise and fidelity graph support.

This module defines ``UnifiedEstimator`` that extends the behaviour of
the original ``FastBaseEstimator`` seed by:
  * accepting a PyTorch ``nn.Module`` and evaluating it on batches of
    parameters;
  * optionally adding Gaussian shot‑noise to mimic finite‑sample
    effects;
  * providing a helper that builds a weighted graph from
    fidelity‑based similarities of the model outputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor of shape (batch, 1) for a single parameter set."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class UnifiedEstimator:
    """Hybrid estimator that evaluates a PyTorch model and optionally adds shot noise.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    noise : bool, optional
        If ``True`` the ``evaluate`` method will add Gaussian noise
        to the deterministic outputs.
    shots : int, optional
        Number of shots to use for the noise simulation; if ``None`` the
        model is evaluated deterministically.
    """

    def __init__(self, model: nn.Module, *, noise: bool = False, shots: Optional[int] = None) -> None:
        self.model = model
        self.noise = noise
        self.shots = shots

    def _evaluate_raw(self, observables: Iterable[ScalarObservable], parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        raw = self._evaluate_raw(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1.0 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def fidelity_adjacency(
        self,
        vectors: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> "nx.Graph":
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(range(len(vectors)))
        for i, vi in enumerate(vectors):
            for j, vj in enumerate(vectors[i + 1 :], i + 1):
                fid = (
                    (vi / (vi.norm() + 1e-12))
                   .dot(vj / (vj.norm() + 1e-12))
                   .abs()
                   .item()
                    ** 2
                )
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def FCL(n_features: int = 1) -> nn.Module:
        """Return a simple fully‑connected PyTorch layer mimicking the quantum example."""
        class FullyConnectedLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(n_features, 1)

            def forward(self, thetas: Iterable[float]) -> torch.Tensor:
                vals = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
                return torch.tanh(self.linear(vals)).mean(dim=0)

        return FullyConnectedLayer()
