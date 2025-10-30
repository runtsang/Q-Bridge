import torch
import torch.nn as nn
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor


class GraphQNN(nn.Module):
    """
    Classical graph neural network that mirrors the GraphQNN seed utilities.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. [4, 8, 1].
    device : torch.device, optional
        Device on which to place the network.
    """

    def __init__(self, arch: Sequence[int], device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.arch = list(arch)
        self.device = torch.device(device)
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.Tanh()
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for lin in self.layers:
            h = self.activation(lin(h))
        return h

    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Generate a random feed‑forward network, a synthetic training set,
        and the target weight matrix of the final layer.
        """
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target = weights[-1]
        data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(target.size(1), dtype=torch.float32)
            y = target @ x
            data.append((x, y))
        return list(arch), weights, data, target

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """
        Overlap‑squared between two vectors, normalised to unity.
        """
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph where nodes are samples and edges are
        weighted by state‑fidelity of the feature vectors.
        """
        G = nx.Graph()
        G.add_nodes_from(range(len(samples)))
        for (i, (xi, _)), (j, (xj, _)) in itertools.combinations(enumerate(samples), 2):
            fid = self.state_fidelity(xi, xj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    def train(self, data: Iterable[Tuple[Tensor, Tensor]], lr: float = 1e-3, epochs: int = 200):
        """
        Very small training loop for demonstration purposes.
        Uses MSE loss and Adam optimiser.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                preds = self.forward(x.to(self.device))
                loss = loss_fn(preds, y.to(self.device))
                loss.backward()
                optimizer.step()
