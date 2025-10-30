import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple, List

Tensor = torch.Tensor

class GraphQNN__gen291(nn.Module):
    """
    Classical graph neural network that extends the original GraphQNN.
    Provides a GCN feature extractor, a multi‑output MLP, and a
    layer‑wise loss for training on fidelity targets.
    """

    def __init__(self, qnn_arch: Sequence[int], hidden_dim: int = 32, device: str = "cpu"):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.hidden_dim = hidden_dim
        self.device = device

        # Feature extractor: GCN‑like linear layer
        self.gcn = nn.Linear(self.qnn_arch[0], self.hidden_dim, bias=False)

        # MLP to predict fidelities for each target state
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.qnn_arch[-1]),
        )
        self.to(self.device)

    def forward(self, graph: nx.Graph) -> List[Tensor]:
        """
        Return a list of node embeddings for each layer.
        """
        adjacency = nx.to_numpy_array(graph, dtype=torch.float32)
        node_features = torch.from_numpy(adjacency).to(self.device)

        embeddings = [node_features]
        current = node_features
        for _ in range(1, len(self.qnn_arch)):
            current = self.gcn(current)
            embeddings.append(current)
        return embeddings

    def predict(self, graph: nx.Graph) -> Tensor:
        """
        Predict a vector of fidelities for the given graph.
        """
        embeddings = self.forward(graph)
        last = embeddings[-1]
        preds = self.mlp(last)
        return preds.mean(dim=0)

    def fit(
        self,
        graphs: Iterable[nx.Graph],
        targets: Iterable[Tensor],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """
        Train the network using a layer‑wise MSE loss.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for g, y in zip(graphs, targets):
                optimizer.zero_grad()
                preds = self.predict(g)
                loss = loss_fn(preds, y.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={total_loss / len(graphs):.4f}")

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
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
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen291.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = GraphQNN__gen291.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored
