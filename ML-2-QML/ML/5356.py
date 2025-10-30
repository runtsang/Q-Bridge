"""UnifiedSamplerGraphRegressor – classical side of the hybrid sampler‑regression model.

The design pulls in ideas from all four reference seeds:
* Classical Sampler (SamplerQNN) – a 2‑layer MLP with softmax output.
* Graph‑based feed‑forward (GraphQNN) – uses a random linear layer sequence and fidelity‑based graph construction.
* Quantum regression (QuantumRegression) – a regression head with a linear output.
* Classical‑quantum binary classification (HybridFunction/Hybrid) – a differentiable sigmoid head.

The class exposes a single ``forward`` that returns a probability distribution over two classes
and a regression target.  The loss can be either cross‑entropy or MSE depending on the
target type.  All tensors are **compatible** with PyTorch’s autograd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import List, Tuple, Iterable

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (features, target) pairs for training."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Create a random feed‑forward network and training data."""
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
    """Compute activations for each layer on the provided samples."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: List[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that mimics a quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: Tensor, shift: float) -> Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: Tensor) -> Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class UnifiedSamplerGraphRegressor(nn.Module):
    """Hybrid sampler‑regressor with a graph‑based classical backbone."""
    def __init__(
        self,
        num_features: int,
        qnn_arch: List[int] | None = None,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.shift = shift

        # Build a random feed‑forward backbone (like GraphQNN)
        if qnn_arch is None:
            qnn_arch = [num_features, 64, 32, 2]
        self.arch = qnn_arch
        self.weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f)) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        )

        # Classification head (sampler) – softmax over two outputs
        self.classifier = nn.Linear(qnn_arch[-1], 2)

        # Regression head – a simple linear layer
        self.regressor = nn.Linear(qnn_arch[-1], 1)

        # Hybrid sigmoid layer to mimic quantum expectation
        self.hybrid = Hybrid(qnn_arch[-1], shift=self.shift)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (probabilities, regression_output)."""
        activations = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)

        # Classification probabilities
        logits = self.classifier(activations[-1])
        probs = F.softmax(logits, dim=-1)

        # Regression output
        reg_raw = self.regressor(activations[-1]).squeeze(-1)
        reg = self.hybrid(reg_raw)  # apply sigmoid to keep in [0,1]

        return probs, reg

    def fidelity_graph(self, states: List[Tensor], threshold: float) -> nx.Graph:
        """Convenience wrapper around the graph builder."""
        return fidelity_adjacency(states, threshold)

__all__ = ["UnifiedSamplerGraphRegressor", "HybridFunction", "Hybrid"]
