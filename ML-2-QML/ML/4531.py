import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import Iterable, Tuple, List, Sequence

class QuantumClassifierModel:
    """Hybrid classical model that unifies a standard feed‑forward classifier,
    a QCNN‑style convolutional block, a graph‑based propagation module,
    and a sampler network for probability estimates."""
    def __init__(self, num_features: int = 8, depth: int = 3,
                 graph_arch: Sequence[int] = (8, 16, 8), sampler_dim: int = 4) -> None:
        self.classifier = self._build_classifier(num_features, depth)
        self.qcnn = self._build_qcnn(num_features)
        self.graph_arch = graph_arch
        self.graph_network = self._build_graph_network(graph_arch)
        self.sampler = self._build_sampler(sampler_dim)

    def _build_classifier(self, num_features: int, depth: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def _build_qcnn(self, num_features: int) -> nn.Module:
        class QCNNModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_map = nn.Sequential(nn.Linear(num_features, 2 * num_features), nn.Tanh())
                self.conv1 = nn.Sequential(nn.Linear(2 * num_features, 2 * num_features), nn.Tanh())
                self.pool1 = nn.Sequential(nn.Linear(2 * num_features, num_features), nn.Tanh())
                self.conv2 = nn.Sequential(nn.Linear(num_features, num_features), nn.Tanh())
                self.pool2 = nn.Sequential(nn.Linear(num_features, num_features // 2), nn.Tanh())
                self.head = nn.Linear(num_features // 2, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                x = self.feature_map(x)
                x = self.conv1(x)
                x = self.pool1(x)
                x = self.conv2(x)
                x = self.pool2(x)
                return torch.sigmoid(self.head(x))

        return QCNNModule()

    def _build_graph_network(self, arch: Sequence[int]) -> nn.Module:
        class GraphModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers = []
                for in_f, out_f in zip(arch[:-1], arch[1:]):
                    layers.append(nn.Linear(in_f, out_f))
                    layers.append(nn.Tanh())
                layers.append(nn.Linear(arch[-1], 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return self.net(x)

        return GraphModule()

    def _build_sampler(self, dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(2, dim),
            nn.Tanh(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clf_out = self.classifier(x)
        qcnn_out = self.qcnn(x)
        graph_out = self.graph_network(x)
        sampler_out = self.sampler(x)
        return clf_out, qcnn_out, graph_out, sampler_out

    def fidelity_adjacency(self, states: Iterable[torch.Tensor], threshold: float,
                           secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = torch.dot(a / (torch.norm(a) + 1e-12), b / (torch.norm(b) + 1e-12)).item() ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["QuantumClassifierModel"]
