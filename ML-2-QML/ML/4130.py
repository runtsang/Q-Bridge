"""Hybrid classical convolutional / graph neural network module.

The class ConvHybridGen107 can be used as a drop‑in replacement for the
original Conv module.  It supports:
* a traditional 2‑D convolutional filter (torch.nn.Conv2d)
* a graph‑based neural network that operates on flattened patches
  and uses state‑fidelity to build adjacency graphs
* a lightweight estimator that evaluates arbitrary scalar observables
  on batches of inputs

The implementation is fully classical and relies only on PyTorch
and NetworkX.  It is deliberately lightweight so that it can be used
in resource‑constrained environments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic input‑target pairs for a linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random feed‑forward network and a training set."""
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a classical MLP and return all layer activations."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two unit‑norm tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class ConvHybridGen107(nn.Module):
    """Hybrid classical convolution / graph neural network.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square patch that is convolved over the input.
    threshold : float, default 0.0
        Activation threshold for the classical convolution.
    qnn_arch : Sequence[int] | None, default None
        Architecture of a graph‑based network.  If ``None`` the module
        behaves as a plain convolution.
    use_graph : bool, default False
        When ``True`` the network is built from ``qnn_arch`` and the
        convolution step is replaced by a graph‑based feedforward.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        qnn_arch: Sequence[int] | None = None,
        use_graph: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_graph = use_graph

        if use_graph:
            if qnn_arch is None:
                raise ValueError("qnn_arch must be provided when use_graph=True")
            self.arch, self.weights, self.training_data, self.target_weight = random_network(
                qnn_arch, samples=10
            )
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_graph:
            # Flatten to 1‑D and run the graph network
            tensor = x.flatten()
            activations = feedforward(self.arch, self.weights, [(tensor, None)])
            return activations[-1]
        else:
            tensor = x.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations

    def run(self, data: Tensor | list[list[float]]) -> float:
        """Apply the chosen filter to ``data`` and return a scalar."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        output = self.forward(tensor)
        return output.mean().item()

    def fidelity_graph(self, threshold: float) -> nx.Graph:
        """Return a graph built from the current layer outputs."""
        if not self.use_graph:
            raise RuntimeError("fidelity_graph is only available for graph networks")
        # Run a single forward pass to collect states
        states = []
        for tensor, _ in self.training_data:
            states.append(
                feedforward(self.arch, self.weights, [(tensor, None)])[0][-1]
            )
        return fidelity_adjacency(states, threshold)

    # ------------------------------------------------------------------
    # Estimator utilities
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a list of observables on batches of parameters.

        The method mirrors FastBaseEstimator.evaluate from the original
        repository but is fully classical.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def Conv() -> ConvHybridGen107:
    """Factory returning a default hybrid filter."""
    return ConvHybridGen107()

__all__ = ["ConvHybridGen107", "Conv"]
