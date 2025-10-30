import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Tuple, Sequence, List
import itertools
import networkx as nx

# ----------------------------------------------------------------------
# Graph‑based utilities (from reference 4)
# ----------------------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    stored = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----------------------------------------------------------------------
# Fully‑connected layer (from reference 2)
# ----------------------------------------------------------------------
class FCL:
    """Simple fully‑connected layer with a run method."""
    def __init__(self, n_features: int = 1):
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# ----------------------------------------------------------------------
# Classical circuit builder (from reference 1)
# ----------------------------------------------------------------------
def build_classifier_circuit_classical(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
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

def build_classifier_circuit(num_features: int, depth: int, mode: str = "classical") -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Dispatch to the appropriate circuit builder."""
    if mode == "classical":
        return build_classifier_circuit_classical(num_features, depth)
    raise NotImplementedError("Quantum circuit construction is only available in the QML module.")

# ----------------------------------------------------------------------
# Hybrid function and layer (from reference 3)
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.inputs = inputs
        logits = inputs + shift
        outputs = torch.sigmoid(logits)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Simple hybrid head that replaces a quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

# ----------------------------------------------------------------------
# Main model (combining all ideas)
# ----------------------------------------------------------------------
class QuantumClassifierModelGen060(nn.Module):
    """
    A unified classifier that can operate in classical, quantum, or hybrid mode.
    Optionally augments features with a graph‑based embedding derived from
    fidelity‑based adjacency.
    """
    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        use_graph: bool = False,
        graph_arch: Sequence[int] | None = None,
        mode: str = "classical",
        backend=None,
        shots: int = 1024,
    ):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.mode = mode
        self.use_graph = use_graph
        self.graph_arch = graph_arch or [num_features, num_features, 2]
        self.backend = backend
        self.shots = shots

        if mode == "classical":
            self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth, mode="classical")
        elif mode == "hybrid":
            self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth, mode="classical")
            self.hybrid = Hybrid(num_features, shift=np.pi / 2)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        if use_graph:
            self.graph_arch, self.unitaries, self.training_data, self.target_unitary = random_network(self.graph_arch, samples=10)
            self.graph_embedding = self._compute_graph_embedding()

    def _compute_graph_embedding(self) -> torch.Tensor:
        """Create a simple embedding from the unitary layers."""
        embeddings = []
        for layer_ops in self.unitaries[1:]:
            for op in layer_ops:
                state = op * self.target_unitary
                vec = torch.tensor(state.full().flatten(), dtype=torch.float32)
                embeddings.append(vec)
        return torch.mean(torch.stack(embeddings), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_graph:
            x = x + self.graph_embedding
        if self.mode == "classical":
            logits = self.network(x)
            probs = F.softmax(logits, dim=-1)
            return probs
        elif self.mode == "hybrid":
            features = self.network(x)
            probs = self.hybrid(features)
            return probs

__all__ = [
    "FCL",
    "build_classifier_circuit",
    "HybridFunction",
    "Hybrid",
    "QuantumClassifierModelGen060",
]
