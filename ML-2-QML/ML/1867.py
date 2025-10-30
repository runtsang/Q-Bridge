import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple

class GraphQNNGen052:
    """Hybrid graph neural network with classical feature extractor.

    The class mirrors the original GraphQNN interface but adds a
    trainable MLP that transforms raw features before they are passed
    through the classical linear layers that represent the “quantum”
    part in the seed.  A ``train`` method is provided that can
    optimise the MLP and the linear layers jointly or separately.
    """

    def __init__(self,
                 qnn_arch: Sequence[int],
                 hidden_dim: int | None = None,
                 device: str = "cpu") -> None:
        self.arch = list(qnn_arch)
        self.device = device
        self.hidden_dim = hidden_dim or qnn_arch[1]
        self.pre = nn.Sequential(
            nn.Linear(self.arch[0], self.hidden_dim, bias=False),
            nn.Tanh(),
        )
        self.weights = nn.ParameterList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.weights.append(nn.Parameter(
                torch.randn(out_f, in_f, dtype=torch.float32)))
        self.to(device)

    @staticmethod
    def random_training_data(weight: torch.Tensor,
                             samples: int,
                             seed: int | None = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(seed)
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int],
                       samples: int,
                       seed: int | None = None) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        rng = np.random.default_rng(seed)
        weights = [torch.randn(out_f, in_f, dtype=torch.float32)
                   for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        target_weight = weights[-1]
        training_data = GraphQNNGen052.random_training_data(target_weight, samples, seed)
        return list(qnn_arch), weights, training_data, target_weight

    def feedforward(self,
                    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        stored: List[List[torch.Tensor]] = []
        for features, _ in samples:
            x = self.pre(features.to(self.device))
            activations = [x]
            current = x
            for w in self.weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen052.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(self,
              samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
              epochs: int = 200,
              lr: float = 1e-3,
              train_quantum: bool = True,
              verbose: bool = False) -> None:
        params = list(self.pre.parameters())
        if train_quantum:
            params += list(self.weights)
        opt = optim.Adam(params, lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            opt.zero_grad()
            loss = 0.0
            for x, y in samples:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.feedforward([(x, y)])
                loss += loss_fn(outputs[-1][0], y)
            loss /= len(samples)
            loss.backward()
            opt.step()
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss.item():.4f}")
