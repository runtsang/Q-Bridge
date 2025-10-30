import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Sequence, Iterable, List, Tuple

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1))
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]):
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        act = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            act.append(current)
        activations.append(act)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: List[Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for i, ai in enumerate(states):
        for j, aj in enumerate(states[i+1:], i+1):
            fid = state_fidelity(ai, aj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
    return G

class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: List[Tensor], b: List[Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, shift: float) -> Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

class Hybrid(nn.Module):
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: Tensor) -> Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class GraphQNNHybrid:
    """
    Classical graph‑based neural network with an RBF kernel and a hybrid sigmoid head.
    """
    def __init__(self, qnn_arch: Sequence[int], shift: float = 0.0, samples: int = 10):
        self.arch = list(qnn_arch)
        self.shift = shift
        self.arch, self.weights, self.training_data, self.target_weight = random_network(self.arch, samples)
        self.classifier = None

    def feedforward(self, samples: List[Tuple[Tensor, Tensor]]):
        return feedforward(self.arch, self.weights, samples)

    def fidelity_adjacency(self, states: List[Tensor], threshold: float,
                           *, secondary: float | None = None):
        return fidelity_adjacency(states, threshold, secondary=secondary)

    def kernel_matrix(self, samples: List[Tuple[Tensor, Tensor]], gamma: float = 1.0) -> np.ndarray:
        activations = [state[-1] for state in self.feedforward(samples)]
        return kernel_matrix(activations, activations, gamma=gamma)

    def train_classifier(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        self.classifier = LogisticRegression()
        self.classifier.fit(X, y)

    def predict(self, X: np.ndarray):
        if self.classifier is None:
            raise RuntimeError("Classifier not trained")
        return self.classifier.predict(X)

__all__ = ["GraphQNNHybrid", "HybridFunction", "Hybrid", "KernalAnsatz", "Kernel", "kernel_matrix"]
