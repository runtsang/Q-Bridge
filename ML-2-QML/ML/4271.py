"""Hybrid classical module providing a classifier, sampler, and graph utilities.

The interface mirrors the quantum implementation: ``build_classifier_circuit`` returns a
PyTorch ``nn.Module`` together with metadata used by the quantum helper.  The
classical sampler implements a soft‑max neural network that can be used in lieu
of a quantum sampler.  Graph utilities compute fidelity‑based adjacency graphs
from state vectors, enabling fidelity‑guided regularisation or data augmentation.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import networkx as nx
from torch import Tensor

# --------------------------------------------------------------------------- #
# 1. Classical classifier
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    use_residual: bool = False,
    res_scale: float = 0.5,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], nn.Linear]:
    """
    Construct a feed‑forward classifier that mirrors the quantum ansatz.

    Parameters
    ----------
    num_features:
        Number of input features / qubits.
    depth:
        Number of hidden layers.
    use_residual:
        Add a residual shortcut between the input and each hidden layer.
    res_scale:
        Scaling factor for the residual contribution.

    Returns
    -------
    network:
        PyTorch sequential model.
    encoding:
        List of feature indices that are directly fed into the network
        (identical to the quantum encoding indices).
    weight_sizes:
        Number of trainable parameters in each layer, used for bookkeeping.
    head:
        Final linear layer producing logits for the two‑class problem.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    # Hidden layers
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features, bias=True)
        layers.extend([linear, nn.ReLU(inplace=True)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    # Residual shortcut (if requested)
    if use_residual:
        # A simple identity mapping scaled by res_scale
        class Residual(nn.Module):
            def __init__(self, scale: float):
                super().__init__()
                self.scale = scale

            def forward(self, x: Tensor) -> Tensor:
                return x * self.scale

        layers.append(Residual(res_scale))

    # Head
    head = nn.Linear(in_dim, 2, bias=True)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    return network, encoding, weight_sizes, head


# --------------------------------------------------------------------------- #
# 2. Classical sampler
# --------------------------------------------------------------------------- #

def SamplerQNN() -> nn.Module:
    """
    Simple two‑layer neural network that outputs a probability distribution
    over two classes using a soft‑max function.

    Returns
    -------
    nn.Module
        A model with shape ``(2,) -> (4,) -> (2,)``.
    """
    class _Sampler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4, bias=True),
                nn.Tanh(),
                nn.Linear(4, 2, bias=True),
            )

        def forward(self, inputs: Tensor) -> Tensor:
            return F.softmax(self.net(inputs), dim=-1)

    return _Sampler()


# --------------------------------------------------------------------------- #
# 3. Graph utilities (adapted from GraphQNN)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight


def feedforward(
    qnn_arch: List[int],
    weights: List[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
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
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Iterable[Tensor],
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
    "build_classifier_circuit",
    "SamplerQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
