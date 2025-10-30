import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import itertools
from typing import Iterable, List, Tuple

# ----------------------------------------------------------------------
#  Graph‑based QNN utilities (classical)
# ----------------------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Random dense layer weights."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_network(qnn_arch: List[int], samples: int):
    """Generate a random QNN architecture and synthetic training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    dataset = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1))
        target = target_weight @ features
        dataset.append((features, target))
    return qnn_arch, weights, dataset, target_weight

def feedforward(qnn_arch: List[int], weights: List[torch.Tensor],
                samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
    """Forward pass through the QNN."""
    activations = []
    for features, _ in samples:
        layer_out = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_out.append(current)
        activations.append(layer_out)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Iterable[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# ----------------------------------------------------------------------
#  Classical RBF kernel
# ----------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: List[torch.Tensor], b: List[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
#  Hybrid classifier (classical)
# ----------------------------------------------------------------------
class ClassicalHybridClassifier(nn.Module):
    """
    CNN backbone + graph‑based QNN + classical RBF kernel read‑out.
    The graph QNN provides a low‑dimensional embedding that is mapped
    into a high‑dimensional feature space by the RBF kernel, after which
    a linear classifier produces the final logits.
    """

    def __init__(self,
                 in_channels: int = 3,
                 graph_arch: List[int] = [10, 20, 5],
                 gamma: float = 1.0,
                 training_samples: int = 200):
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Graph‑based QNN
        self.qnn_arch = graph_arch
        self.qnn_weights, self.qnn_train_states = self._init_qnn(training_samples)

        # Kernel read‑out
        self.kernel = Kernel(gamma)
        self.kernel_classifier = nn.Linear(len(self.qnn_train_states), 2)

    def _init_qnn(self, samples: int):
        _, weights, training_data, _ = random_network(self.qnn_arch, samples)
        # Compute the final state for each training sample
        train_states = [feedforward(self.qnn_arch, weights,
                                    [(feat, None)])[0][-1] for feat, _ in training_data]
        return weights, train_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extractor
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Graph‑based QNN state
        final_state = feedforward(self.qnn_arch, self.qnn_weights,
                                  [(x, None)])[0][-1]

        # Kernel similarity to training states
        K = kernel_matrix([final_state], self.qnn_train_states)
        K_tensor = torch.tensor(K, dtype=torch.float32)

        # Linear read‑out on the kernel vector
        out = self.kernel_classifier(K_tensor)
        return out

__all__ = ["ClassicalHybridClassifier"]
