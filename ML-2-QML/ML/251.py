import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from typing import Iterable, Sequence, List, Tuple

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

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
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNGen:
    """
    Hybrid graph‑based neural network that can run on a CPU or GPU.

    The class mirrors the original GraphQNN interface but adds:
    1. A PyTorch nn.Module that implements the same feed‑forward semantics.
    2. A simple regulariser that penalises the fidelity between pairs of hidden states.
    3. Automatic device selection (CPU or CUDA GPU).
    """

    def __init__(self, arch: Sequence[int], device: str | None = None):
        self.arch = list(arch)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.model.to(self.device)

    def _build_model(self) -> nn.Sequential:
        layers = [nn.Linear(self.arch[i], self.arch[i + 1]) for i in range(len(self.arch) - 1)]
        return nn.Sequential(*layers)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        activations = []
        for features, _ in samples:
            act = [features.to(self.device)]
            x = features.to(self.device)
            for layer in self.model:
                x = torch.tanh(layer(x))
                act.append(x)
            activations.append(act)
        return activations

    def fidelity_regulariser(self, activations: List[List[Tensor]]) -> Tensor:
        """
        Compute a regularisation term that penalises high fidelity
        between any pair of hidden activations.
        """
        reg = torch.tensor(0.0, device=self.device)
        for sample_acts in activations[1:]:  # exclude input layer
            for i in range(len(sample_acts) - 1):
                for j in range(i + 1, len(sample_acts)):
                    a = sample_acts[i]
                    b = sample_acts[j]
                    fid = state_fidelity(a, b)
                    reg += fid ** 2
        return reg

    def train(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
        epochs: int,
        lr: float = 1e-3,
        reg_weight: float = 0.0,
    ) -> None:
        """
        Simple training loop using MSE loss and optional fidelity regularisation.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for features, target in samples:
                optimizer.zero_grad()
                activations = self.feedforward([(features, target)])
                pred = activations[0][-1]
                loss = nn.functional.mse_loss(pred, target.to(self.device))
                if reg_weight > 0.0:
                    reg = self.fidelity_regulariser(activations)
                    loss += reg_weight * reg
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
