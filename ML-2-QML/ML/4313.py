import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import networkx as nx

class SamplerQNNGen327(nn.Module):
    """Hybrid classical sampler inspired by QCNN and GraphQNN.

    The network first extracts features with a QCNN‑style feed‑forward
    encoder.  A similarity graph is then built from the hidden
    representation and a lightweight graph‑neural‑network layer
    aggregates messages along the edges.  The final layer produces a
    probability distribution over two classes.
    """
    def __init__(self, graph_threshold: float = 0.9):
        super().__init__()
        self.threshold = graph_threshold
        # QCNN‑style feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh()
        )
        # Graph message‑passing block
        self.graph_block = nn.Sequential(
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        h = self.encoder(x)
        # Build cosine‑similarity graph (proxy for fidelity)
        norm = h / (h.norm(dim=-1, keepdim=True) + 1e-12)
        sim = torch.matmul(norm, norm.t())
        adj = (sim >= self.threshold).float()
        # Message passing: weighted sum over neighbours
        h = torch.matmul(adj, h)
        # Produce logits
        logits = self.graph_block(h)
        return F.softmax(logits, dim=-1)

# ---------------------------------------------------------------------------

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic (input, target) pairs for a linear model."""
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_graph_network(qnn_arch: list[int], samples: int):
    """Create a random linear network that mimics the GraphQNN architecture.

    Returns the architecture, a list of weight matrices, the training data
    and the target weight (the last layer).
    """
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight

def feedforward_graph(qnn_arch: list[int], weights: list[torch.Tensor], samples: list[tuple[torch.Tensor, torch.Tensor]]):
    """Run a forward pass through a purely linear network."""
    activations = []
    for features, _ in samples:
        layer_outs = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_outs.append(current)
        activations.append(layer_outs)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two pure state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float, *,
                       secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build an undirected graph where edges represent high fidelity."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

__all__ = [
    "SamplerQNNGen327",
    "random_graph_network",
    "feedforward_graph",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
]
