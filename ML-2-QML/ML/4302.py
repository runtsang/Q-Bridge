import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import List, Tuple, Iterable, Sequence

Tensor = torch.Tensor


class HybridSamplerQNN(nn.Module):
    """
    A hybrid sampler network that combines the classical SamplerQNN,
    a lightweight GraphQNN utility set, and a small feed‑forward
    classifier.  The class exposes three API groups:

    1. **Sampling** – a two‑layer softmax network that mirrors the
       original SamplerQNN.
    2. **Graph utilities** – random network generation, forward
       propagation, state fidelity and adjacency graph construction.
    3. **Classification** – a modular classifier that can be built
       on demand, reusing the same architecture pattern as the
       quantum counterpart.

    The implementation is deliberately compact but fully
    importable and type‑annotated.
    """

    def __init__(
        self,
        sampler_arch: Sequence[int] = (2, 4, 2),
        classifier_depth: int = 2,
        graph_arch: Sequence[int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.sampler_net = self._build_sampler(sampler_arch)
        self.classifier_depth = classifier_depth
        self.graph_arch = graph_arch

    # ------------------------------------------------------------------
    # 1. Classical sampler
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sampler(arch: Sequence[int]) -> nn.Sequential:
        layers: List[nn.Module] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(arch[-1], arch[-1]))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Return a probability distribution over the two output classes."""
        return F.softmax(self.sampler_net(x), dim=-1)

    def sample(self, x: Tensor) -> Tensor:
        """Convenience wrapper around the forward pass."""
        return self.forward(x)

    # ------------------------------------------------------------------
    # 2. Graph utilities
    # ------------------------------------------------------------------
    @staticmethod
    def random_network(
        qnn_arch: Sequence[int], samples: int = 10
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a toy feed‑forward network and a small training set."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f))
        target_weight = weights[-1]
        training_data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1))
            target = target_weight @ features
            training_data.append((features, target))
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Run a forward pass through the toy network."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for w in weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared inner product between two unit‑normed vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a weighted graph where edges encode state fidelity."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(
            enumerate(states), 2
        ):
            fid = HybridSamplerQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # 3. Classification
    # ------------------------------------------------------------------
    def build_classifier(
        self, num_features: int, depth: int | None = None
    ) -> nn.Module:
        """Return a simple feed‑forward classifier network."""
        if depth is None:
            depth = self.classifier_depth
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def classify(self, x: Tensor) -> Tensor:
        """Classify an input using a freshly built classifier."""
        clf = self.build_classifier(x.shape[-1])
        return clf(x)


__all__ = ["HybridSamplerQNN"]
