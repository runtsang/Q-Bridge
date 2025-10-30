"""Unified hybrid QNN in classical mode.

This module merges the four seed families into a single interface.  The
``HybridQNN`` class can be instantiated in one of four modes:
* ``estimator`` – lightweight feed‑forward regressor,
* ``conv``      – 2‑D convolutional filter,
* ``classifier`` – multi‑layer MLP with metadata,
* ``graph``     – graph‑based QNN utilities.

The same class is re‑exported by convenience functions for backward
compatibility.

Typical usage:

```python
from UnifiedEstimatorQNN import HybridQNN, EstimatorQNN, Conv, QuantumClassifier, GraphQNN

est = EstimatorQNN()
conv = Conv(kernel_size=3, threshold=0.1)
clf = QuantumClassifier(num_features=5, depth=3)
graph = GraphQNN(qnn_arch=[3,4,5], samples=200)
```

"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

__all__ = ["HybridQNN", "EstimatorQNN", "Conv", "QuantumClassifier", "GraphQNN"]


# --------------------------------------------------------------------------- #
# 1. Classical Estimator (feed‑forward regressor)                              #
# --------------------------------------------------------------------------- #
def _build_estimator() -> nn.Module:
    """Return a small fully‑connected network similar to the original seed."""
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    )


# --------------------------------------------------------------------------- #
# 2. Classical convolutional filter                                           #
# --------------------------------------------------------------------------- #
class _ConvFilter(nn.Module):
    """Drop‑in replacement for a quantum filter using a 2‑D convolution."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution and return the mean activation."""
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # add batch and channel
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))  # mean over spatial dims


# --------------------------------------------------------------------------- #
# 3. Classical classifier                                                     #
# --------------------------------------------------------------------------- #
def _build_classifier(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Build a simple MLP that mimics the original quantum classifier interface."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# 4. Graph‑based QNN utilities                                               #
# --------------------------------------------------------------------------- #
def _random_linear(in_feats: int, out_feats: int) -> torch.Tensor:
    """Return a random weight matrix with the requested shape."""
    return torch.randn(out_feats, in_feats, dtype=torch.float32)


def _random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic data that maps features to targets via a linear transform."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def _random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Create a random feed‑forward network and a target weight for supervised training."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = _random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def _feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Forward‑propagate a batch of samples through the network."""
    activations: List[List[torch.Tensor]] = []
    for features, _ in samples:
        layer_out = features
        layerwise = [layer_out]
        for weight in weights:
            layer_out = torch.tanh(weight @ layer_out)
            layerwise.append(layer_out)
        activations.append(layerwise)
    return activations


def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared overlap of two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def _fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                        *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph based on state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 5. Unified class                                                            #
# --------------------------------------------------------------------------- #
class HybridQNN:
    """Unified estimator that can operate in four distinct modes.

    Parameters
    ----------
    mode : {'estimator', 'conv', 'classifier', 'graph'}
        The functional mode of the instance.
    backend : str, optional
        Only used in the quantum sub‑module; defaults to ``'cpu'``.
    **kwargs : dict
        Additional arguments specific to the chosen mode.
    """

    def __init__(self, mode: str, backend: str = 'cpu', **kwargs):
        self.mode = mode
        self.backend = backend
        self._registry: List[str] = []

        if mode == 'estimator':
            self.model = _build_estimator()
            for name, _ in self.model.named_parameters():
                self._registry.append(name)

        elif mode == 'conv':
            kernel_size = kwargs.get('kernel_size', 2)
            threshold = kwargs.get('threshold', 0.0)
            self.conv = _ConvFilter(kernel_size=kernel_size, threshold=threshold)

        elif mode == 'classifier':
            num_features = kwargs.get('num_features', 4)
            depth = kwargs.get('depth', 2)
            self.network, self.encoding, self.weight_sizes, self.observables = _build_classifier(
                num_features, depth
            )
            for name, _ in self.network.named_parameters():
                self._registry.append(name)

        elif mode == 'graph':
            qnn_arch = kwargs.get('qnn_arch', [3, 4, 5])
            samples = kwargs.get('samples', 200)
            self.arch, self.weights, self.training_data, self.target_weight = _random_network(
                qnn_arch, samples
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def forward(self, inputs):
        """Run the network (or filter) on the provided inputs."""
        if self.mode == 'estimator':
            return self.model(inputs)
        elif self.mode == 'conv':
            return self.conv(inputs)
        elif self.mode == 'classifier':
            return self.network(inputs)
        else:
            raise NotImplementedError("Forward not defined for graph mode.")

    def run(self, inputs):
        """Alias for ``forward`` to match the original EstimatorQNN API."""
        return self.forward(inputs)

    def parameter_names(self) -> List[str]:
        """Return all registered parameter names."""
        return list(self._registry)

    # --------------------------------------------------------------------- #
    # Graph utilities (only available in graph mode)
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        return _random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        return _feedforward(qnn_arch, weights, samples)

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        return _state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return _fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


# --------------------------------------------------------------------------- #
# Convenience wrappers for backward compatibility                           #
# --------------------------------------------------------------------------- #
def EstimatorQNN(**kwargs):
    """Return a HybridQNN instance in estimator mode."""
    return HybridQNN(mode='estimator', **kwargs)

def Conv(**kwargs):
    """Return a HybridQNN instance in conv mode."""
    return HybridQNN(mode='conv', **kwargs)

def QuantumClassifier(**kwargs):
    """Return a HybridQNN instance in classifier mode."""
    return HybridQNN(mode='classifier', **kwargs)

def GraphQNN(**kwargs):
    """Return a HybridQNN instance in graph mode."""
    return HybridQNN(mode='graph', **kwargs)
