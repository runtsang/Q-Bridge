import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple, List

# --------------------------------------------------------------------------- #
#  Classical RBF kernel (depth‑controlled)
# --------------------------------------------------------------------------- #
class ClassicalRBF(nn.Module):
    """Depth‑controlled radial basis function kernel."""
    def __init__(self, gamma: float = 1.0, depth: int = 1):
        super().__init__()
        self.gamma = gamma
        self.depth = depth

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        base = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        return base ** self.depth

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    depth: int = 1
) -> np.ndarray:
    """Return the Gram matrix for two sets of feature vectors."""
    kern = ClassicalRBF(gamma, depth)
    return np.array([[kern(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Graph‑based QNN utilities (classical analogue)
# --------------------------------------------------------------------------- #
def random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_network(
    qnn_arch: Sequence[int],
    samples: int
) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate random linear weights and synthetic training data."""
    weights = [random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = [(torch.randn(target_weight.size(1)), target_weight @ torch.randn(target_weight.size(1))) for _ in range(samples)]
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]
) -> List[List[torch.Tensor]]:
    """Forward pass through a purely linear network."""
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
    """Squared overlap of two classical state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Hybrid head (purely classical)
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    """Simple linear head with differentiable sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x.view(x.size(0), -1))
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

# --------------------------------------------------------------------------- #
#  UnifiedKernelClassifier
# --------------------------------------------------------------------------- #
class UnifiedKernelClassifier(nn.Module):
    """
    A classifier that can operate in three modes:
        * 'classical' – uses ClassicalRBF kernel and a simple feed‑forward net.
        * 'hybrid'    – uses the HybridHead on top of a dense backbone.
        * 'quantum'   – not implemented in this module; see qml_code.
    """
    def __init__(self, mode: str = 'classical', **kwargs):
        super().__init__()
        self.mode = mode.lower()
        if self.mode == 'classical':
            self.kernel = ClassicalRBF(kwargs.get('gamma', 1.0), kwargs.get('depth', 1))
            self.backbone = nn.Sequential(
                nn.Linear(kwargs.get('input_dim', 10), kwargs.get('hidden_dim', 20)),
                nn.ReLU(),
                nn.Linear(20, 2)
            )
        elif self.mode == 'hybrid':
            self.head = HybridHead(kwargs.get('input_dim', 10), kwargs.get('shift', 0.0))
        else:
            raise NotImplementedError("Quantum mode requires the quantum module.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'classical':
            return self.backbone(x)
        elif self.mode == 'hybrid':
            return self.head(x)
        else:
            raise RuntimeError("Unsupported mode.")
