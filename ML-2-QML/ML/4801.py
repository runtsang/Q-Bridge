"""Classical hybrid binary classifier that emulates the quantum architecture.

This module provides a fully classical implementation that mirrors the
quantum counterpart. It uses a classical convolutional filter inspired
by the quanvolution example, dense layers, and a graph‑based feature
aggregation based on state fidelities. The model is fully
PyTorch‑compatible and can be used as a baseline for the quantum
version."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Classical hybrid head (logistic sigmoid)                                 #
# --------------------------------------------------------------------------- #
class ClassicalHybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation mimicking the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class ClassicalHybrid(nn.Module):
    """Linear head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return ClassicalHybridFunction.apply(self.linear(logits), self.shift)

# --------------------------------------------------------------------------- #
# 2. Classical quanvolution filter                                            #
# --------------------------------------------------------------------------- #
class ClassicalQuanvolutionFilter(nn.Module):
    """Classical 2×2 convolutional filter mimicking the quantum quanvolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# 3. Main classifier                                                         #
# --------------------------------------------------------------------------- #
class HybridQuantumBinaryClassifier(nn.Module):
    """Classical hybrid binary classifier mirroring the quantum architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.fc1 = nn.Linear(4 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = ClassicalHybrid(in_features=1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qfilter(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

# --------------------------------------------------------------------------- #
# 4. Graph‑based utilities (state fidelity & adjacency)                     #
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def feedforward(qnn_arch: list[int], weights: list[torch.Tensor],
                samples: list[tuple[torch.Tensor, torch.Tensor]]) -> list[list[torch.Tensor]]:
    stored = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1))
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    weights = [torch.randn(out, in_) for in_, out in zip(qnn_arch[:-1], qnn_arch[1:])]
    training_data = random_training_data(weights[-1], samples)
    return qnn_arch, weights, training_data, weights[-1]

__all__ = [
    "ClassicalHybridFunction",
    "ClassicalHybrid",
    "ClassicalQuanvolutionFilter",
    "HybridQuantumBinaryClassifier",
    "state_fidelity",
    "fidelity_adjacency",
    "feedforward",
    "random_network",
    "random_training_data",
]
