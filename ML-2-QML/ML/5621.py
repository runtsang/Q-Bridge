"""Hybrid estimator that mirrors the quantum‑classical architecture.

The module implements:
* Classical neural network construction (mirroring the quantum `build_classifier_circuit`).
* Random network utilities and fidelity‑based graph construction from GraphQNN.
* A `HybridFunction` that emulates a quantum expectation head using a sigmoid.
* `HybridEstimatorQNN` – a PyTorch module that exposes the same interface as its quantum counterpart.

The design keeps the ML side completely classical (PyTorch) but retains the same API so that one can swap the implementation for the quantum version without changing downstream code."""
from __future__ import annotations

import torch
import torch.nn as nn
import itertools
import networkx as nx
from typing import Iterable, Sequence, Tuple, List

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Random weight generation utilities (GraphQNN-inspired)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic training set for a linear transformation."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random feed‑forward network and a training set for its last layer."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Propagate a batch of samples through a random network."""
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
    """Return the squared overlap between two vectors."""
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
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Classical classifier construction (mirrors QuantumClassifierModel.py)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a small feed‑forward classifier and expose its metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
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
# Hybrid head emulating a quantum expectation
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that mimics a quantum expectation layer."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

# --------------------------------------------------------------------------- #
# Main estimator module
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN(nn.Module):
    """Classical estimator that mirrors the hybrid quantum network."""
    def __init__(self, input_dim: int, arch: Sequence[int], depth: int, shift: float = 0.0, device: str = "cpu") -> None:
        super().__init__()
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(input_dim, depth)
        self.shift = shift
        self.device = device
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(inputs)                       # shape: (batch, 2)
        probs = HybridFunction.apply(logits[:, 0], self.shift)  # use first logit as proxy
        return torch.stack((probs, 1 - probs), dim=-1)

    def fidelity_graph(self, samples: Iterable[Tuple[Tensor, Tensor]], threshold: float) -> nx.Graph:
        """Generate a fidelity‑based adjacency graph from a random network."""
        arch, weights, _, _ = random_network([len(self.encoding), *self.weight_sizes], len(list(samples)))
        activations = feedforward(arch, weights, samples)
        states = [act[-1] for act in activations]
        return fidelity_adjacency(states, threshold)
