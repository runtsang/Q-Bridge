from __future__ import annotations

import itertools
import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Callable, List, Optional
import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def _apply_noise(values: List[float], shots: int | None, seed: int | None) -> List[float]:
    if shots is None:
        return values
    rng = np.random.default_rng(seed)
    return [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in values]

def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def _build_fidelity_graph(states: Sequence[torch.Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class ConvFilter(nn.Module):
    """Simple 2D convolution filter mimicking a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class FastBaseEstimatorGen:
    """Hybrid estimator that evaluates a PyTorch model with optional shot noise,
    convolution filtering and fidelity-based graph analysis."""
    def __init__(self,
                 model: nn.Module,
                 conv_filter: Optional[ConvFilter] = None) -> None:
        self.model = model
        self.conv_filter = conv_filter

    def evaluate(self,
                 observables: Iterable[ScalarObservable] | None = None,
                 parameter_sets: Sequence[Sequence[float]] = (),
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]
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
                row = _apply_noise(row, shots, seed)
                results.append(row)
        return results

    def fidelity_graph(self,
                       results: List[List[float]],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        states = [torch.tensor(r, dtype=torch.float32) for r in results]
        return _build_fidelity_graph(states, threshold,
                                     secondary=secondary,
                                     secondary_weight=secondary_weight)

__all__ = ["FastBaseEstimatorGen", "ConvFilter"]
