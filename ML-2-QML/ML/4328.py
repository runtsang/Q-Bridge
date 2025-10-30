import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Sequence

# --------------------------------------------------------------------------- #
# 1. Hybrid classical filter – can delegate to a quantum circuit if supplied
# --------------------------------------------------------------------------- #
class HybridConvFilter(nn.Module):
    """
    A drop‑in replacement for a single 2×2 convolutional layer that can
    optionally delegate computation to a user‑supplied quantum circuit.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel (default 2).
    threshold : float
        Activation threshold used when the classical path is active.
    use_quantum : bool
        If True, the provided `quantum_circuit` is used for inference.
    quantum_circuit : object
        Any callable with a `.run(data)` method that accepts a 2‑D array.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 use_quantum: bool = False,
                 quantum_circuit: object | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.quantum_circuit = quantum_circuit

        if not use_quantum:
            # Classical 1‑in/1‑out conv with bias
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor | List[float] | List[List[float]]) -> float:
        """
        Forward pass.

        If `use_quantum` is True, delegates to the quantum circuit.
        Otherwise applies the classical convolution and returns the mean
        sigmoid activation over the kernel window.
        """
        if self.use_quantum and self.quantum_circuit is not None:
            return self.quantum_circuit.run(data)

        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# --------------------------------------------------------------------------- #
# 2. Classical utilities – graph‑based state analysis
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic (x, Wx) pairs for a given linear map."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Build a random classical MLP with the given architecture and a training set
    that targets the last layer's linear map.
    """
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[torch.Tensor],
                samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Compute activations of every layer for a batch of samples."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> 'nx.Graph':
    """Build a weighted graph from state fidelities."""
    import networkx as nx
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
# 3. Classifier factory – identical API for classical and quantum
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a simple feed‑forward classifier.

    Returns
    -------
    network : nn.Module
        Sequential network consisting of `depth` blocks of Linear+ReLU
        followed by a 2‑class head.
    encoding : List[int]
        Indices of input features used for encoding (identity here).
    weight_sizes : List[int]
        Number of trainable parameters per linear layer.
    observables : List[int]
        Dummy observable list to mirror the quantum API.
    """
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

__all__ = [
    "HybridConvFilter",
    "build_classifier_circuit",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
